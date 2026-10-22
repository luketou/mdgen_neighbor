# Issue tracker: GitHub

Issues and specs for this repository live as GitHub Issues in `luketou/mdgen_neighbor`. Prefer the connected GitHub tools when available; the `gh` CLI is also supported. If `gh` is unavailable or unauthenticated, use the connector fallback in [tools](tools.md). Always select this repository explicitly.

## Spec and ticket mapping

- One spec maps to exactly one GitHub Issue.
- One GitHub Issue may contain multiple tickets.
- Represent tickets as Markdown checkboxes in the Issue body, with each ticket describing a bounded implementation scope.
- Keep the spec's background, goal, acceptance criteria, and all related tickets together in that Issue.

## Conventions

- **Create an issue**: `gh issue create --repo luketou/mdgen_neighbor --title "..." --body "..."`.
- **Read an issue**: `gh issue view <number> --repo luketou/mdgen_neighbor --comments`, including its labels.
- **List issues**: use `gh issue list --repo luketou/mdgen_neighbor` with appropriate state and label filters.
- **Comment on an issue**: `gh issue comment <number> --repo luketou/mdgen_neighbor --body "..."`.
- **Apply or remove labels**: `gh issue edit <number> --repo luketou/mdgen_neighbor --add-label "..."` or `--remove-label "..."`.
- **Close an issue**: `gh issue close <number> --repo luketou/mdgen_neighbor --comment "..."`.

Issue titles and bodies must follow the Traditional Chinese writing rules in [conventions](conventions.md).

## Pull requests as a triage surface

**PRs as a request surface: no.**

GitHub shares one number space across issues and pull requests. Resolve an ambiguous `#<number>` with the corresponding PR and issue read tools, or `gh pr view <number> --repo luketou/mdgen_neighbor` followed by `gh issue view <number> --repo luketou/mdgen_neighbor`.

## Skill operations

- When a skill says **publish to the issue tracker**, create one GitHub Issue for the spec and put its tickets in that Issue body.
- When a skill says **fetch the relevant ticket**, read the parent GitHub Issue and select the named or numbered ticket from its body.

## Wayfinding operations

- **Map**: one Issue labelled `wayfinder:map`, containing Notes, Decisions-so-far, and Fog.
- **Child ticket**: normally a checkbox ticket inside the map Issue. Create a separate child Issue only when independent assignment or dependency tracking requires it, and link it back to the map.
- **Blocking**: use GitHub native issue dependencies for separate child Issues. For checkbox tickets, record `Blocked by: Ticket <n>` in the ticket text.
- **Frontier query**: choose the first unchecked, unassigned ticket whose blockers are complete.
- **Claim**: assign the Issue to the driving developer and identify the active ticket in a comment.
- **Resolve**: check the ticket in the Issue body and record the result in a comment. Close the Issue only after every ticket and the spec acceptance criteria are complete.
