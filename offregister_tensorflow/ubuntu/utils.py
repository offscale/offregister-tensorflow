from fabric.contrib.files import exists
from offregister_fab_utils.apt import apt_depends
from offregister_fab_utils.fs import cmd_avail
from offregister_fab_utils.git import clone_or_update
from offutils import update_d


def instantiate_virtual_env(c, pip_version, python3, virtual_env, virtual_env_args):
    virtual_env_dir = virtual_env[: virtual_env.rfind("/")]
    if not exists(c, runner=c.run, path=virtual_env_dir) or not exists(
        c, runner=c.run, path=virtual_env
    ):
        c.sudo('mkdir -p "{virtual_env_dir}"'.format(virtual_env_dir=virtual_env_dir))
        user_group = c.run("echo $(id -un):$(id -gn)", hide=True).stdout.rstrip()
        c.sudo(
            'chown -R {user_group} "{virtual_env_dir}"'.format(
                user_group=user_group, virtual_env_dir=virtual_env_dir
            )
        )
        if python3:
            if pip_version:
                c.sudo(
                    "pip3 install pip=={pip_version}".format(pip_version=pip_version)
                )
            else:
                c.sudo("pip3 install -U pip setuptools")
            # `--system-site-packages` didn't install a pip
            c.run(
                'python3 -m venv "{virtual_env}" {virtual_env_args}'.format(
                    virtual_env=virtual_env, virtual_env_args=virtual_env_args or ""
                )
            )

            c.sudo("pip3 install keras_applications==1.0.6 --no-deps")
            c.sudo("pip3 install keras_preprocessing==1.0.5 --no-deps")
            c.sudo("pip3 install h5py==2.8.0")
        else:
            c.sudo("pip2 install pip=={pip_version}".format(pip_version=pip_version))
            if not cmd_avail(c, "virtualenv"):
                c.sudo("pip2 install virtualenv")
            c.run(
                'virtualenv --system-site-packages "{virtual_env}"'.format(
                    virtual_env=virtual_env
                )
            )
            c.sudo("pip2 install keras_applications==1.0.6 --no-deps")
            c.sudo("pip2 install keras_preprocessing==1.0.5 --no-deps")
            c.sudo("pip2 install h5py==2.8.0")


def setup_gpu(download_dir):
    if not cmd_avail(c, "nvidia-smi"):
        c.run("mkdir -p {download_dir}".format(download_dir=download_dir))
        with c.cd(download_dir):
            c.sudo(
                "apt-key adv --fetch-keys "
                "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub"
            )
            repo_pkg = "nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb"
            for pkg, url in (
                (
                    "cuda-repo-ubuntu1804_10.0.130-1_amd64.deb",
                    "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/",
                ),
                (
                    repo_pkg,
                    "http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/",
                ),
            ):
                c.run("curl -OL {url}{pkg}".format(url=url, pkg=pkg))
                c.sudo("apt install ./{pkg}".format(pkg=pkg))
            c.sudo("apt update")
            c.sudo("apt install ./{repo_pkg}".format(repo_pkg=repo_pkg))
            c.sudo("apt-get install -y --no-install-recommends nvidia-driver-418")
        raise NotImplementedError("You must restart machine for NVIDIA driver to work")
    c.run("nvidia-smi")
    c.sudo(
        "apt-get install -y --no-install-recommends "
        "cuda-10-0 "
        "libcudnn7=7.6.2.24-1+cuda10.0 "
        "libcudnn7-dev=7.6.2.24-1+cuda10.0"
    )
    c.sudo(
        "apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda10.0 "
        "libnvinfer-dev=5.1.5-1+cuda10.0"
    )
    apt_depends(c, "clang")
    return dict(
        LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64",
        TF_NEED_CUDA="1",
        TF_CUDA_CLANG="1",
        TF_NEED_TENSORRT="1",
        TF_CUDA_COMPUTE_CAPABILITIES="3.7",  # TODO: Check device against NVIDIA website list of supported CUDA versions
        CLANG_CUDA_COMPILER_PATH="/usr/bin/clang"
        # TF_DOWNLOAD_CLANG='1'
    )


def build_from_source(
    repo_dir,
    force_rebuild,
    tensorflow_tag,
    tensorflow_branch,
    build_env,
    gpu,
    use_sudo,
    python3,
    run_cmd,
    virtual_env,
    extra_build_args="",
):
    apt_depends(c, "unzip")
    c.run("pip uninstall -y tensorflow", warn=True, hide=True)
    c.run("mkdir -p {repo_dir}".format(repo_dir=repo_dir))

    tf_repo = "{repo_dir}/{repo}".format(
        repo_dir=repo_dir, repo="tensorflow-for-py3" if python3 else "tensorflow"
    )
    clone_or_update(
        repo="tensorflow",
        team="tensorflow",
        to_dir=tf_repo,
        skip_reset=True,
        skip_checkout=True,
        use_sudo=use_sudo,
        **(
            {"branch": tensorflow_branch}
            if tensorflow_branch
            else {"tag": tensorflow_tag}
        )
    )
    with c.cd(tf_repo):
        version = 3 if python3 else 2
        processor = "gpu" if gpu else "cpu"  # 'tpu'
        release_to = "{repo_dir}/tensorflow_pkg-{version}-{processor}".format(
            repo_dir=repo_dir, version=version, processor=processor
        )
        if force_rebuild:
            run_cmd("rm -rf '{}'".format(release_to))

        whl = "{release_to}/*cp{version}-{processor}*.whl".format(
            release_to=release_to, version=version, processor=processor
        )
        if exists(c, runner=c.run, path=release_to) and exists(
            c, runner=c.run, path=whl
        ):
            return "Already built"

        if python3:
            run_cmd("python -m pip install numpy wheel")

        run_cmd("python -m pip install keras_preprocessing")

        # install_bazel()
        if not cmd_avail(c, "bazel"):
            with c.cd(
                "{parent}/Downloads".format(
                    parent=c.run(
                        "dirname {repo_dir}".format(repo_dir=repo_dir), hide=True
                    )
                )
            ):
                script = "bazel-0.25.2-installer-linux-x86_64.sh"
                c.run(
                    "curl -OL https://github.com/bazelbuild/bazel/releases/download/0.25.2/"
                    "{script}".format(script=script)
                )
                c.sudo("bash {script}".format(script=script))
        default_build_env = dict(
            PYTHON_BIN_PATH="{}/bin/python".format(virtual_env),
            PYTHON_LIB_PATH=virtual_env,
            TF_DOWNLOAD_MKL="1",
            TF_NEED_MKL="1",
            CC_OPT_FLAGS="-march=native",
            TF_NEED_JEMALLOC="1",
            TF_NEED_GCP="0",
            TF_NEED_HDFS="0",
            TF_ENABLE_XLA="0",  # JIT
            TF_NEED_VERBS="0",
            TF_NEED_OPENCL="0",
            TF_NEED_OPENCL_SYCL="0",
            TF_NEED_COMPUTECPP="0",
            TF_NEED_CUDA="0",
            TF_NEED_MPI="0",
            TF_NEED_S3="0",
            TF_NEED_GDR="0",
            TF_SET_ANDROID_WORKSPACE="0",
            TF_NEED_KAFKA="0",
            TF_CUDA_CLANG="0",
            TF_DOWNLOAD_CLANG="0",
            TF_NEED_ROCM="0",
            TF_NEED_TENSORRT="0",
        )
        env = dict(**update_d(default_build_env, build_env))
        c.run("env", env=env)
        c.run("./configure", env=env)
        c.run(
            "bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package {}".format(
                extra_build_args
            )
        )
        c.run(
            "bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag {}".format(
                release_to
            )
        )
    return whl
