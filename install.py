import launch
import sys
import os
import shutil
import subprocess
from importlib import metadata
from packaging.version import parse

dir_repos = "repositories"
script_path = os.path.dirname(os.path.abspath(__file__))

python = sys.executable
git = os.environ.get('GIT', "git")

# Whether to default to printing command output
default_command_live = (os.environ.get('WEBUI_LAUNCH_LIVE_OUTPUT') == "1")

def repo_dir(name):
    return os.path.join(script_path, dir_repos, name)

def git_fix_workspace(dir, name):
    launch.run(f'"{git}" -C "{dir}" fetch --refetch --no-auto-gc', f"Fetching all contents for {name}", f"Couldn't fetch {name}", live=True)
    launch.run(f'"{git}" -C "{dir}" gc --aggressive --prune=now', f"Pruning {name}", f"Couldn't prune {name}", live=True)
    return

def run_git(dir, name, command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live, autofix=True):
    try:
        return launch.run(f'"{git}" -C "{dir}" {command}', desc=desc, errdesc=errdesc, custom_env=custom_env, live=live)
    except RuntimeError:
        if not autofix:
            raise

    print(f"{errdesc}, attempting autofix...")
    git_fix_workspace(dir, name)

    return launch.run(f'"{git}" -C "{dir}" {command}', desc=desc, errdesc=errdesc, custom_env=custom_env, live=live)

def git_clone(url, dir, name, commithash=None):
    # TODO clone into temporary dir and move if successful

    if os.path.exists(dir):
        if commithash is None:
            return

        current_hash = run_git(dir, name, 'rev-parse HEAD', None, f"Couldn't determine {name}'s hash: {commithash}", live=False).strip()
        if current_hash == commithash:
            return

        if run_git(dir, name, 'config --get remote.origin.url', None, f"Couldn't determine {name}'s origin URL", live=False).strip() != url:
            run_git(dir, name, f'remote set-url origin "{url}"', None, f"Failed to set {name}'s origin URL", live=False)

        run_git(dir, name, 'fetch', f"Fetching updates for {name}...", f"Couldn't fetch {name}", autofix=False)

        run_git(dir, name, f'checkout {commithash}', f"Checking out commit for {name} with hash: {commithash}...", f"Couldn't checkout commit {commithash} for {name}", live=True)

        return

    try:
        launch.run(f'"{git}" clone "{url}" "{dir}"', f"Cloning {name} into {dir}...", f"Couldn't clone {name}", live=True)
    except RuntimeError:
        shutil.rmtree(dir, ignore_errors=True)
        raise

    if commithash is not None:
        launch.run(f'"{git}" -C "{dir}" checkout {commithash}', None, "Couldn't checkout {name}'s hash: {commithash}")

def get_installed_version(package: str):
    try:
        return metadata.version(package)
    except Exception:
        return None

def extract_base_package(package_string: str) -> str:
    base_package = package_string.split("@git")[0]
    return base_package

def install_requirements(file_path='requirements.txt'):
    try:
        # Read the requirements file
        with open(file_path, 'r') as file:
            requirements = file.read().splitlines()

        # Install each requirement
        for requirement in requirements:
            package = requirement.strip()
            if "==" in package:
                package_name, package_version = package.split("==")
                installed_version = get_installed_version(package_name)
                if installed_version != package_version:
                    launch.run_pip(
                        f'install -U "{package}"',
                        f"openvino extension requirement: changing {package_name} version from {installed_version} to {package_version}",
                    )
            elif ">=" in package:
                package_name, package_version = package.split(">=")
                installed_version = get_installed_version(package_name)
                if not installed_version or parse(
                    installed_version
                ) < parse(package_version):
                    launch.run_pip(
                        f'install -U "{package}"',
                        f"openvino extension  requirement: changing {package_name} version from {installed_version} to {package_version}",
                    )
            elif "<=" in package:
                package_name, package_version = package.split("<=")
                installed_version = get_installed_version(package_name)
                if not installed_version or parse(
                    installed_version
                ) > parse(package_version):
                    launch.run_pip(
                        f'install "{package_name}=={package_version}"',
                        f"openvino extension  requirement: changing {package_name} version from {installed_version} to {package_version}",
                    )
            elif not launch.is_installed(extract_base_package(package)):
                launch.run_pip(
                    f'install "{package}"',
                    f"openvino extension  requirement: {package}",
                )
            
        
        print("OpenVINO extension: All requirements installed successfully.")
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
    except Exception as e:
        print(e)
        print(f"Warning: Failed to install {package}, some preprocessors may not work.")

def install():

    install_requirements()
    
    print("OpenVINO extension install complete")


install()
