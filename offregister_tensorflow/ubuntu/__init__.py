from functools import partial
from os import environ

from fabric.api import run, sudo
from fabric.context_managers import shell_env
from fabric.contrib.files import exists
from fabric.operations import _run_command
from offregister_fab_utils.apt import apt_depends
from offregister_fab_utils.ubuntu.systemd import restart_systemd, install_upgrade_service
from offregister_opencv.base import dl_install_opencv

from offregister_tensorflow.ubuntu.utils import build_from_source, setup_gpu, instantiate_virtual_env


def install_tensorflow0(python3=False, virtual_env=None, virtual_env_args=None, *args, **kwargs):
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
    instantiate_virtual_env(pip_version, python3, virtual_env, virtual_env_args)

    if not exists(virtual_env):
        raise ReferenceError('Virtualenv does not exist')

    with shell_env(VIRTUAL_ENV=virtual_env, PATH="{}/bin:$PATH".format(virtual_env)):
        if pip_version:
            run('pip install pip=={pip_version}'.format(pip_version=pip_version))
        else:
            run('pip install -U pip setuptools')
        run('pip install -U jupyter')

        if kwargs.get('from') == 'source':
            gpu = kwargs.get('GPU')
            if gpu:
                setup_gpu(download_dir='{home}/Downloads'.format(home=home))

            whl = build_from_source(repo_dir='{home}/repos'.format(home=home), gpu=gpu,
                                    tensorflow_tag=kwargs.get('tensorflow_tag', 'v1.14.0'),
                                    force_rebuild=kwargs.get('force_rebuild'), use_sudo=kwargs.get('use_sudo'),
                                    python3=python3, run_cmd=run_cmd, virtual_env=virtual_env)
            if whl.endswith('.whl'):
                run('pip install {whl}'.format(whl=whl))
        elif kwargs.get('from') == 'pypi' or 'from' not in kwargs:
            run('pip install -U tensorflow')

        if kwargs.get('skip_tflow_example'):
            return run('python -c "from tensorflow import __version__;'
                       ' print(\'Installed TensorFlow {} successfully\'.format(__version__))"')

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

    return install_upgrade_service(
        'jupyter_notebook',
        conf_local_filepath=kwargs.get('systemd-conf-file'),
        context={
            'ExecStart': ' '.join((
                '{virtual_env}/bin/jupyter'.format(virtual_env=virtual_env),
                'notebook',
                "--NotebookApp.notebook_dir='{notebook_dir}'".format(notebook_dir=notebook_dir),
                "--NotebookApp.ip='{listen_ip}'".format(listen_ip=kwargs.get('listen_ip', '127.0.0.1')),
                "--NotebookApp.port='{listen_port}'".format(listen_port=kwargs.get('listen_port', '8888')),
                "--Session.username='{User}'".format(User=user),
                "--NotebookApp.password='{password}'".format(password=environ['PASSWORD']),
                "--NotebookApp.password_required='True'",
                "--NotebookApp.allow_remote_access='True'",
                "--NotebookApp.iopub_data_rate_limit='2147483647'",
                '--no-browser',
                "--NotebookApp.open_browser='False'"
            )),
            'Environments': kwargs['Environments'],
            'WorkingDirectory': notebook_dir,
            'User': user,
            'Group': group
        }
    )


def install_opencv2(virtual_env=None, python3=False, *args, **kwargs):
    home = run('echo $HOME', quiet=True)
    virtual_env = virtual_env or '{home}/venvs/tflow'.format(home=home)

    site_packages = run('{virtual_env}/bin/python -c "import site; print(site.getsitepackages()[0])"'.format(
        virtual_env=virtual_env
    ))

    dl_install_opencv(
        extra_cmake_args='OPENCV_PYTHON3_INSTALL_PATH={site_packages}'.format(site_packages=site_packages)
        if python3
        else "-D PYTHON2_PACKAGES_PATH='{virtual_env}/lib/python2.7/site-packages' "
             "-D PYTHON2_LIBRARY='{virtual_env}/bin'".format(virtual_env=virtual_env)
    )


def install_tensorboard3(extra_opts=None, virtual_env=None, pip_install_args=None, *args, **kwargs):
    home = run('echo $HOME', quiet=True)
    virtual_env = virtual_env or '{home}/venvs/tflow'.format(home=home)
    tensorboard_logs_dir = kwargs.get('tensorboard_logs_dir', '{home}/tensorboard_logs_dir'.format(home=home))
    run('mkdir -p {tensorboard_logs_dir}'.format(tensorboard_logs_dir=tensorboard_logs_dir))

    conf_name = 'tensorboard'

    with shell_env(PATH='{}/bin:$PATH'.format(virtual_env), VIRTUAL_ENV=virtual_env):
        run('pip install {pip_install_args} -U tensorboard'.format(pip_install_args=pip_install_args or ''))

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
                    ('{virtual_env}/bin/tensorboard'.format(virtual_env=virtual_env),
                     "--logdir '{tensorboard_logs_dir}'".format(tensorboard_logs_dir=tensorboard_logs_dir),
                     extra_opts if extra_opts else '')),
                'Environments': kwargs['Environments'],
                'WorkingDirectory': tensorboard_logs_dir,
                'User': kwargs['User'], 'Group': kwargs['Group']
            }
        )
    return restart_systemd(conf_name)
