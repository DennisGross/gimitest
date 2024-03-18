def get_values():
    print("HERREEE!")
    return 1, 2  # Function returns two values

try:
    # Attempt to unpack returned values into three variables
    a, b, c = get_values()
except ValueError as e:
    # Catch the ValueError caused by too many/few variables for unpacking
    print(f"Error: {e}")
    # Get error name
    if str(e).find("not enough values to unpack") == -1:
        print(f"Error: {e}")
    else:
        print("FOUND")