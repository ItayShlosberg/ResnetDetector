import os
import subprocess

def get_folder_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def get_size(package_name):
    process = subprocess.Popen(['pip', 'show', package_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    details = stdout.decode().split('\n')
    for detail in details:
        if detail.startswith('Location:'):
            location = detail.split(': ')[1].strip()
            return get_folder_size(location)
    return 0  # If location is not found, return 0 size

def main():
    print(1)
    process = subprocess.Popen(['pip', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(2)
    stdout, stderr = process.communicate()
    print(3)
    packages = [line.split()[0] for line in stdout.decode().split('\n')[2:] if line]
    print(4)
    sizes = [(package, get_size(package)) for package in packages]
    print(5)
    for package, size in sorted(sizes, key=lambda x: x[1], reverse=True):
        print(f"{package}: {size/1024**2:.2f} MB")  # Convert from bytes to MB

if __name__ == '__main__':
    main()
