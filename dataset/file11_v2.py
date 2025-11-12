def and_gate(input_1: int, input_2: int) -> int:
    """
    Compute the logical AND between two binary inputs.

    >>> and_gate(0, 0)
    0
    >>> and_gate(0, 1)
    0
    >>> and_gate(1, 0)
    0
    >>> and_gate(1, 1)
    1
    """
    result = int(input_1 and input_2)
    print(f"AND({input_1}, {input_2}) = {result}")
    return result


def n_input_and_gate(inputs: list[int]) -> int:
    """
    Compute the AND of all binary inputs in a list.

    >>> n_input_and_gate([1, 0, 1, 1, 0])
    0
    >>> n_input_and_gate([1, 1, 1, 1, 1])
    1
    """
    return int(all(inputs))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    print("All tests completed successfully!")
