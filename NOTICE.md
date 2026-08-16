# Attribution

The initial commit of this repository is the source code of
[abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis),
at commit `d557527793820f1e6b06872256824255facd47fd` (2024-03-22), used as a
starting baseline.

Two modifications were made to that baseline before committing:

1. **Credentials redacted.** The upstream notebooks contain a live Roboflow API
   key and a full Google session cookie header. Both were replaced with
   placeholders rather than republished.
2. **Notebook outputs cleared**, to keep the diff readable.

Everything after the initial commit is my own work. To see exactly what I
changed, diff against the first commit:

```bash
git diff $(git rev-list --max-parents=0 HEAD) HEAD
```

The upstream project is a video tutorial companion repo and carries no license
file. It is credited here in full; if the author requests removal of the
baseline commit, it will be rewritten out of history.
