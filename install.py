import launch
import os



# Whether to default to printing command output
default_command_live = (os.environ.get('WEBUI_LAUNCH_LIVE_OUTPUT') == "1")

def install():

    #install_requirements()
    req = "requirements.txt"
    launch.run(f"pip install -r {req}", f"Install requirements for OpenVINO",
               f"Error: OpenVINO dependencies were not installed correctly, OpenVINO extension will not be enabled", live = default_command_live)
    
    print("OpenVINO extension install complete")


install()
