def check_configuration(parameter):
    """Check C++ specific configurations"""
    parameter = parameter.strip().title()
    configurations = {
        "CompilerFlags": "-std=c++17 -Wall -Wextra",
        "HeaderPaths": ["/usr/include/c++/11", "./include"],
        "StandardLibraries": ["libstdc++.so.6", "libgcc_s.so.1"],
        "MemoryLimit": "2GB",
        "OptimizationLevel": "O2"
    }
    return configurations.get(parameter, "Parameter not found in C++ configuration")

if __name__ == "__main__":
    print(check_configuration("CompilerFlags"))