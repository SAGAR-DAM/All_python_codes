def my_function(greeting, times, *args, **kwargs):
    # Print the greeting message multiple times
    for _ in range(times):
        print(greeting)

    # Extract specific keyword arguments
    name = kwargs.get('name',None)
    age = kwargs.get('age', None)
    city = kwargs.get('city',None)

    # Use the extracted keyword arguments
    if name:
        print(f"Name: {name}")
    if age:
        print(f"Age: {age}")
    if city:
        print(f"City: {city}")

    # Handle cases where the keyword argument is not provided
    country = kwargs.get('country', None)
    print(f"Country: {country}")

    # Handle other arguments if necessary
    print("Other positional arguments:", args)
    print("Other keyword arguments:", {k: v for k, v in kwargs.items() if k not in ['name', 'age', 'city', 'country']})

# Example usage
my_function("Hello!", 7, 1, 2, 3, 4, name="Alice", age=30, city="New York", hobby="Reading")
