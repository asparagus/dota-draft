### Configs
These configs are used by `draft/providers.py` throughout the project.

This information should be private and so after setting it up, make sure to tell git not to track changes on them.

```bash
git update-index --skip-worktree draft/configs/*.yaml
```

### GAR config:
Requires your GAR `location`, `project` and `repository`.

### GCS config:
Requires your GCS `bucket` and `project`.

### Open Dota config:
Optionally will accept your open dota `api_key`, not required.

### Wandb config:
Requires your Wandb `project`.
