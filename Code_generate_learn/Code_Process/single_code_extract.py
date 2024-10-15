def process_generation_to_code(gens):
    if "```python" in gens:
        gens = gens.split("```python")[1].split("```")[0]
    elif "```" in gens:
        gens = gens.split("```")[1].split("```")[0]

    return gens.split("\n")[1:-1]


gens = """
Here is some Python code:
```python
def hello_world():
    print("Hello, world!")
"""

code = process_generation_to_code(gens)

print(code)

result = "\n".join(code)

print(result)
