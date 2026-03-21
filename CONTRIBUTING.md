# Contributing

Thanks for contributing to syltgen.

## Development setup

1. Clone the repository
2. Run setup:

```powershell
.\setup.ps1
```

3. Activate environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Running locally

```powershell
python -m syltgen.main ./input ./output --verbose
```

## Pull request guidelines

- Keep changes focused and small
- Include a clear summary of behavior changes
- Add/adjust docs for user-facing behavior changes
- Prefer root-cause fixes over output-only patches
- Preserve existing CLI behavior unless intentionally changed

## Reporting issues

When filing an issue, include:
- exact command used
- model/device settings
- one problematic file example
- relevant log excerpt (especially traceback)
