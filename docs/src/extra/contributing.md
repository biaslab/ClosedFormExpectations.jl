# Contributing

We welcome contributions to `ClosedFormExpectations.jl`. This page describes how to report bugs, suggest features, and contribute code.

## Reporting Bugs

We track bugs using [GitHub Issues](https://github.com/biaslab/ClosedFormExpectations.jl/issues). Please provide:

- Versions of Julia and `ClosedFormExpectations.jl`
- A minimal reproducible example
- Expected vs. actual behavior

## Contributing Code

### Setup

Use the `dev` command from the Julia package manager to install for development:

```
] dev git@github.com:your_username/ClosedFormExpectations.jl.git
```

### Adding a New (Distribution, Function) Pair

To add a new closed-form expectation:

1. **Identify the distribution and function** for which a closed-form expression exists.
2. **Add the implementation** in the appropriate file under `src/`. Follow the existing directory structure:
   - `src/Normal/` for Normal-related expectations
   - `src/Gamma/` for Gamma-related expectations
   - `src/Exponential/` for Exponential-related expectations
   - `src/MvNormal/` for multivariate Normal expectations
3. **Define a method** for `mean`:
   ```julia
   function mean(::ClosedFormExpectation, f::YourFunctionType, q::YourDistribution)
       # closed-form expression here
   end
   ```
4. **Optionally add the Williams' product**:
   ```julia
   function mean(::ClosedWilliamsProduct, f::YourFunctionType, q::YourDistribution)
       # return SVector of gradients w.r.t. distribution parameters
   end
   ```
5. **Add tests** in the corresponding test file under `test/`. Tests typically validate against Monte Carlo estimates using the Central Limit Theorem.
6. **Update the documentation** in `docs/src/lib/supported-pairs.md`.

### Running Tests

```
] test ClosedFormExpectations
```

### Style Conventions

- Follow the default [Julia style guide](https://docs.julialang.org/en/v1/manual/style-guide/)
- Use 4 spaces for indentation
- Type names use `UpperCamelCase`
- Function names are `lowercase` with underscores where necessary
