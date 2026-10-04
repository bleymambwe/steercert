# CI regression checks

A steering vector or a probe-based monitor validated on one checkpoint can stop
working after a fine-tune, a quantisation or a model update. A saved audit
report turns that into a test.

```bash
steercert demo --out report.json        # synthetic example report
steercert summary report.json           # print a saved report
steercert check report.json \
    --min-efficiency deployed_vector=0.5 \
    --min-ranking 0.7 \
    --min-leverage 0.1
```

`check` exits with status 1 and prints one `FAIL` line per violated threshold.

| Option | Fails when |
|---|---|
| `--min-efficiency NAME=V` | a direction's RMS efficiency is below `V` |
| `--max-efficiency NAME=V` | a direction's RMS efficiency is above `V` (for example a monitor that must stay blind to a channel) |
| `--min-ranking V` | the certificate-to-measurement Spearman is below `V` or missing |
| `--min-leverage V` | absolute leverage $\lVert\bar g\rVert$ is below `V` |
| `--min-stable-rank V` | stable rank is below `V` |
| `--fail-on-warnings` | any warning finding is present |

The same checks are available in Python as `report.check(...)` and
`steercert.check_report(dict, ...)`.

## GitHub Actions sketch

```yaml
- run: python scripts/audit_release_candidate.py --out audit.json
- run: steercert check audit.json --min-efficiency deployed_vector=0.5 --min-ranking 0.7
```
