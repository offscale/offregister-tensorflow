from fabric.api import run, sudo
from fabric.context_managers import cd, shell_env
from fabric.contrib.files import exists
from offregister_fab_utils.apt import apt_depends
from offregister_fab_utils.fs import cmd_avail
from offregister_jupyter.systemd import install_jupyter_notebook_server


def install0(virtual_env=None, *args, **kwargs):
    apt_depends('python2.7', 'python2.7-dev', 'python-dev', 'python-pip', 'python-apt',
                'build-essential', 'sudo', 'git-core', 'libffi-dev', 'libssl-dev',
                'python-software-properties', 'libatlas-dev', 'liblapack-dev')

    if not cmd_avail('virtualenv'):
        sudo('pip install virtualenv')

    home = run('echo $HOME', quiet=True)
    virtual_env = virtual_env or '{home}/venvs/tflow'.format(home=home)

    virtual_env_dir = virtual_env[:virtual_env.rfind('/')]
    if not exists(virtual_env_dir):
        run('mkdir -p "{virtual_env_dir}"'.format(virtual_env_dir=virtual_env_dir), shell_escape=False)
        run('virtualenv --system-site-packages "{virtual_env}"'.format(virtual_env=virtual_env), shell_escape=False)

    with shell_env(VIRTUAL_ENV=virtual_env, PATH="{}/bin:$PATH".format(virtual_env)):
        run('pip install -U tensorflow jupyter')
        hello_world = ''.join(l.strip() for l in '''import tensorflow as tf;
        hello = tf.constant('TensorFlow works!');
        sess = tf.Session();
        print sess.run(hello)
        '''.splitlines())
        run('python -c "{}"'.format(hello_world))

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
        extra_opts="--NotebookApp.password='{password}' --NotebookApp.password_required=True "
                   "--no-browser --NotebookApp.open_browser=False".format(
            password='sha1:<hash>'
        )
    )
