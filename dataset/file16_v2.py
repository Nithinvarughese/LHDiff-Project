import platform

# Determine hosts file location based on OS
system_name = platform.system()
if system_name == "Windows":
    hosts_path = r"C:\Windows\System32\drivers\etc\hosts"
elif system_name == "Linux":
    hosts_path = r"/etc/hosts"
else:
    hosts_path = None

# List of restricted or unwanted websites to be cleaned
blocked_sites = [
    "https://www.sislovesme.com/",
    "https://motherless.com/",
    "https://xhamster.com/",
    "https://www.xnxx.com/",
    "https://www.xvideos.com/",
    "https://www.pornhub.com/"
]

if hosts_path:
    with open(hosts_path, "r+") as hosts_file:
        lines = hosts_file.readlines()
        hosts_file.seek(0)
        for entry in lines:
            if not any(site in entry for site in blocked_sites):
                hosts_file.write(entry)
        hosts_file.truncate()
