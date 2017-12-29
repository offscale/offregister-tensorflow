from os import environ

from fabric.api import run, sudo
from fabric.context_managers import cd, shell_env, settings
from fabric.contrib.files import exists
from offregister_bazel.ubuntu import install_bazel
from offregister_fab_utils.apt import apt_depends
from offregister_fab_utils.fs import cmd_avail
from offregister_fab_utils.git import clone_or_update
from offregister_jupyter.systemd import install_jupyter_notebook_server
from offregister_opencv.base import dl_install_opencv


def install_tensorflow0(python3=False, virtual_env=None, *args, **kwargs):
    apt_depends('build-essential', 'sudo', 'git-core', 'libffi-dev', 'libssl-dev',
                'python-software-properties', 'libatlas-dev', 'liblapack-dev')

    if not cmd_avail('virtualenv'):
        sudo('pip install virtualenv')

    home = run('echo $HOME', quiet=True)
    virtual_env = virtual_env or '{home}/venvs/tflow'.format(home=home)

    if python3:
        apt_depends('python3-numpy', 'python3-dev', 'python3-pip', 'python3-wheel', 'python3-venv')
    else:
        apt_depends('python2.7', 'python2.7-dev', 'python-dev', 'python-pip', 'python-apt',
                    'python-numpy', 'python-wheel')

    virtual_env_dir = virtual_env[:virtual_env.rfind('/')]
    if not exists(virtual_env_dir) or not exists(virtual_env):
        run('mkdir -p "{virtual_env_dir}"'.format(virtual_env_dir=virtual_env_dir), shell_escape=False)
        if python3:
            run('pip3 install -U pip')
            # `--system-site-packages` didn't install a pip
            run('python3 -m venv "{virtual_env}"'.format(virtual_env=virtual_env),
                shell_escape=False)
        else:
            run('pip install -U pip')
            run('virtualenv --system-site-packages "{virtual_env}"'.format(virtual_env=virtual_env), shell_escape=False)

    if not exists(virtual_env):
        raise ReferenceError('Virtualenv does not exist')

    with shell_env(VIRTUAL_ENV=virtual_env, PATH="{}/bin:$PATH".format(virtual_env)):
        run('pip install -U jupyter pip')
        if kwargs.get('from') == 'source':
            run('pip uninstall -y tensorflow', warn_only=True, quiet=True)
            run('mkdir -p repos')
            with cd('repos'):
                tf_repo = 'tensorflow-for-py3' if python3 else 'tensorflow'
                clone_or_update(repo='tensorflow', team='tensorflow', branch=kwargs.get('branch', 'r1.4'),
                                to_dir=tf_repo,
                                skip_reset=True, skip_checkout=False)
                with cd(tf_repo):
                    release_to = '{home}/repos/tensorflow_pkg'.format(home=home)
                    if kwargs.get('force_rebuild'):
                        run('rm -rf {}'.format(release_to))

                    whl = '{release_to}/*cp{version}*.whl'.format(release_to=release_to,
                                                                  version=3 if python3 else 2)
                    if not exists(release_to) or not exists(whl):
                        if python3:
                            run('pip install numpy wheel')

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
                                       TF_SET_ANDROID_WORKSPACE='0'):
                            run('./configure')
                        run(
                            'bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package --incompatible_load_argument_is_label=false')
                        run('bazel-bin/tensorflow/tools/pip_package/build_pip_package {}'.format(release_to))
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
    run("mkdir -p '{notebook_dir}'".format(notebook_dir=notebook_dir))

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


def install_tensorboard3(virtual_env=None, *args, **kwargs):
    home = run('echo $HOME', quiet=True)
    virtual_env = virtual_env or '{home}/venvs/tflow'.format(home=home)
