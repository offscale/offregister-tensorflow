from functools import partial
from os import environ

from fabric.api import run, sudo
from fabric.context_managers import cd, shell_env
from fabric.contrib.files import exists
from fabric.operations import _run_command

from offregister_bazel.ubuntu import install_bazel
from offregister_fab_utils.apt import apt_depends
from offregister_fab_utils.fs import cmd_avail
from offregister_fab_utils.git import clone_or_update
from offregister_fab_utils.ubuntu.systemd import restart_systemd, install_upgrade_service
from offregister_jupyter.systemd import install_jupyter_notebook_server
from offregister_opencv.base import dl_install_opencv


def install_tensorflow0(python3=False, virtual_env=None, *args, **kwargs):
    apt_depends('build-essential', 'sudo', 'git', 'libffi-dev', 'libssl-dev',
                'software-properties-common', 'libatlas-base-dev')

    home = run('echo $HOME', quiet=True)
    virtual_env = virtual_env or '{home}/venvs/tflow'.format(home=home)

    if python3:
        apt_depends('python3-numpy', 'python3-dev', 'python3-pip', 'python3-wheel', 'python3-venv')
    else:
        apt_depends('python2.7', 'python2.7-dev', 'python-dev', 'python-pip', 'python-apt',
                    'python-numpy', 'python-wheel')

    run_cmd = partial(_run_command, sudo=kwargs.get('use_sudo'))

    pip_version = kwargs.get('pip_version', False)
    virtual_env_dir = virtual_env[:virtual_env.rfind('/')]
    if not exists(virtual_env_dir) or not exists(virtual_env):
        sudo('mkdir -p "{virtual_env_dir}"'.format(virtual_env_dir=virtual_env_dir), shell_escape=False)
        user_group = run('echo $(id -un):$(id -gn)', quiet=True)
        sudo('chown -R {user_group} "{virtual_env_dir}"'.format(user_group=user_group, virtual_env_dir=virtual_env_dir),
             shell_escape=False)
        if python3:
            if pip_version:
                sudo('pip3 install pip=={pip_version}'.format(pip_version=pip_version))
            else:
                sudo('pip3 install -U pip setuptools')
            # `--system-site-packages` didn't install a pip
            run('python3 -m venv "{virtual_env}"'.format(virtual_env=virtual_env),
                shell_escape=False)

            sudo('pip3 install keras_applications==1.0.6 --no-deps')
            sudo('pip3 install keras_preprocessing==1.0.5 --no-deps')
            sudo('pip3 install h5py==2.8.0')
        else:
            sudo('pip2 install pip=={pip_version}'.format(pip_version=pip_version))
            if not cmd_avail('virtualenv'):
                sudo('pip2 install virtualenv')
            run('virtualenv --system-site-packages "{virtual_env}"'.format(virtual_env=virtual_env), shell_escape=False)
            sudo('pip2 install keras_applications==1.0.6 --no-deps')
            sudo('pip2 install keras_preprocessing==1.0.5 --no-deps')
            sudo('pip2 install h5py==2.8.0')

    if not exists(virtual_env):
        raise ReferenceError('Virtualenv does not exist')

    with shell_env(VIRTUAL_ENV=virtual_env, PATH="{}/bin:$PATH".format(virtual_env)):
        if pip_version:
            run('pip install pip=={pip_version}'.format(pip_version=pip_version))
        else:
            run('pip install -U pip setuptools')
        run('pip install -U jupyter')
        if kwargs.get('from') == 'source':
            run('pip uninstall -y tensorflow', warn_only=True, quiet=True)
            run('mkdir -p repos')
            with cd('repos'):
                tf_repo = 'tensorflow-for-py3' if python3 else 'tensorflow'
                clone_or_update(repo='tensorflow', team='tensorflow',
                                branch=kwargs.get('tensorflow_tag', 'v2.0.0'),
                                to_dir=tf_repo, skip_reset=True, skip_checkout=True, use_sudo=kwargs.get('use_sudo'))
                with cd(tf_repo):
                    release_to = '{home}/repos/tensorflow_pkg'.format(home=home)
                    if kwargs.get('force_rebuild'):
                        run_cmd('rm -rf {}'.format(release_to))

                    whl = '{release_to}/*cp{version}*.whl'.format(release_to=release_to,
                                                                  version=3 if python3 else 2)
                    if not exists(release_to) or not exists(whl):
                        if python3:
                            run_cmd('pip install numpy wheel')

                        run_cmd('pip install keras_preprocessing')

                        install_bazel()
                        with shell_env(PYTHON_BIN_PATH='{}/bin/python'.format(virtual_env),
                                       PYTHON_LIB_PATH=virtual_env,
                                       TF_DOWNLOAD_MKL='1',
                                       TF_NEED_MKL='1',
                                       CC_OPT_FLAGS='-march=native',
                                       TF_NEED_JEMALLOC='1',
                                       TF_NEED_GCP='0',
                                       TF_NEED_HDFS='0',
                                       TF_ENABLE_XLA='0',  # JIT
                                       TF_NEED_VERBS='0',
                                       TF_NEED_OPENCL='0',
                                       TF_NEED_OPENCL_SYCL='0',
                                       TF_NEED_COMPUTECPP='0',
                                       TF_NEED_CUDA='0',
                                       TF_NEED_MPI='0',
                                       TF_NEED_S3='0',
                                       TF_NEED_GDR='0',
                                       TF_SET_ANDROID_WORKSPACE='0',
                                       TF_NEED_KAFKA='0',
                                       TF_CUDA_CLANG='0',
                                       TF_DOWNLOAD_CLANG='0',
                                       TF_NEED_ROCM='0'):
                            run('./configure')
                        run('bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package')
                        run('bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag {}'.format(release_to))
                    run('pip install {whl}'.format(whl=whl))
        elif kwargs.get('from') == 'pypi' or 'from' not in kwargs:
            run('pip install -U tensorflow')
        hello_world = ''.join(l.strip() for l in '''import tensorflow as tf;
        hello = tf.constant('TensorFlow works!');
        sess = tf.Session();
        print(sess.run(hello))
        '''.splitlines())
        return run('python -c "{}"'.format(hello_world))


def install_jupyter_notebook1(virtual_env=None, *args, **kwargs):
    home = run('echo $HOME', quiet=True)
    virtual_env = virtual_env or '{home}/venvs/tflow'.format(home=home)
    user, group = (lambda ug: (ug[0], ug[1]) if len(ug) > 1 else (ug[0], ug[0]))(
        run('''printf '%s\t%s' "$USER" "$GROUP"''', quiet=True, shell_escape=False).split('\t'))
    notebook_dir = kwargs.get('notebook_dir', '{home}/notebooks'.format(home=home))
    (sudo if kwargs.get('use_sudo') else run)("mkdir -p '{notebook_dir}'".format(notebook_dir=notebook_dir))

    return install_jupyter_notebook_server(
        pythonpath=virtual_env,
        notebook_dir=notebook_dir,
        listen_ip='0.0.0.0',  # kwargs['public_ipv4'],
        listen_port=int(kwargs.get('listen_port', '8888')),
        Environments='Environment=VIRTUAL_ENV={virtual_env}\n'
                     'Environment=PYTHONPATH={virtual_env}'.format(virtual_env=virtual_env),
        User=user, Group=group,
        extra_opts=' '.join(("--NotebookApp.password='{password}'".format(password=environ['PASSWORD']),
                             '--NotebookApp.password_required=True',
                             '--NotebookApp.iopub_data_rate_limit=2147483647',  # send output for longer
                             '--no-browser', '--NotebookApp.open_browser=False'))
    )


def install_opencv2(virtual_env=None, *args, **kwargs):
    home = run('echo $HOME', quiet=True)
    virtual_env = virtual_env or '{home}/venvs/tflow'.format(home=home)
    dl_install_opencv(extra_cmake_args="-D PYTHON2_PACKAGES_PATH='{virtual_env}/lib/python2.7/site-packages' "
                                       "-D PYTHON2_LIBRARY='{virtual_env}/bin'".format(virtual_env=virtual_env))


def install_tensorboard3(extra_opts=None, virtual_env=None, *args, **kwargs):
    run_cmd = partial(_run_command, sudo=kwargs.get('use_sudo'))
    home = run('echo $HOME', quiet=True)
    virtual_env = virtual_env or '{home}/venvs/tflow'.format(home=home)
    notebook_dir = kwargs.get('notebook_dir', '{home}/notebooks'.format(home=home))

    conf_name = 'tensorboard'

    with shell_env(PATH='{}/bin:$PATH'.format(virtual_env), VIRTUAL_ENV=virtual_env):
        run('pip install -U tensorboard')

        '''
        listen_ip, notebook_dir, pythonpath,
        listen_port='8888', conf_name='jupyter_notebook',
        extra_opts=None, **kwargs)
        '''
        install_upgrade_service(
            conf_name,
            conf_local_filepath=kwargs.get('systemd-conf-file'),
            context={
                'ExecStart': ' '.join(
                    ('{pythonpath}/bin/jupyter notebook'.format(pythonpath=virtual_env),
                     "--NotebookApp.notebook_dir='{notebook_dir}'".format(notebook_dir=notebook_dir),
                     '--NotebookApp.ip={listen_ip}'.format(listen_ip=kwargs.get('notebook_ip', '0.0.0.0')),
                     '--NotebookApp.port={listen_port}'.format(listen_port=int(kwargs.get('listen_port', '8888'))),
                     '--Session.username={User}'.format(User=kwargs['User']),
                     extra_opts if extra_opts else '')),
                'Environments': kwargs['Environments'],
                'WorkingDirectory': virtual_env,
                'User': kwargs['User'], 'Group': kwargs['Group']
            }
        )
    return restart_systemd(conf_name)
