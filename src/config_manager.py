def check_configuration(parameter):
    """Check the current configuration for a parameter."""
    configurations = {
        "NetworkBufferSize": 4096,
        "ThreadPoolSize": 8,
    }
    return configurations.get(parameter, "Parameter not found.")

if __name__ == "__main__":
    # Example usage
    parameter = "NetworkBufferSize"
    value = check_configuration(parameter)
    print(f"Current value for {parameter}: {value}")