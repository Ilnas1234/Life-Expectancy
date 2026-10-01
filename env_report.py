import platform, sys, os, subprocess, importlib


def cpu_model():
    if platform.system() == "Windows":
        try:
            out = subprocess.run(["wmic", "cpu", "get", "Name"], capture_output=True, text=True).stdout
            lines = [l.strip() for l in out.splitlines() if l.strip() and l.strip() != "Name"]
            if lines: return lines[0]
        except Exception:
            pass
        try:
            out = subprocess.run(["powershell", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True).stdout.strip()
            if out: return out
        except Exception:
            pass
    if os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"): return line.split(":", 1)[1].strip()
    if platform.system() == "Darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return platform.processor() or "unknown"


def ram_gb():
    try:
        import psutil
        return f"{psutil.virtual_memory().total / 1024**3:.1f} GB"
    except Exception:
        pass
    if platform.system() == "Windows":
        try:
            import ctypes
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS(); m.dwLength = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return f"{m.ullTotalPhys / 1024**3:.1f} GB"
        except Exception:
            pass
    if os.path.exists("/proc/meminfo"):
        kb = int(open("/proc/meminfo").readline().split()[1]); return f"{kb / 1024**2:.1f} GB"
    return "unknown"


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return out or "none detected (not used: all computations on CPU)"
    except Exception:
        return "none detected (not used: all computations on CPU)"


def main():
    print("Processor          :", cpu_model())
    print("Logical CPUs       :", os.cpu_count())
    print("RAM                :", ram_gb())
    print("GPU                :", gpu())
    print("Operating system   :", platform.platform(), "|", platform.version())
    print("Python             :", sys.version.split()[0], platform.python_implementation(), platform.architecture()[0])
    for mod in ["sklearn", "numpy", "pandas", "scipy", "shap", "matplotlib", "seaborn", "statsmodels", "openpyxl"]:
        try:
            m = importlib.import_module(mod); print(f"{mod:<19}:", getattr(m, "__version__", "?"))
        except Exception:
            print(f"{mod:<19}: not installed")


if __name__ == "__main__":
    main()
