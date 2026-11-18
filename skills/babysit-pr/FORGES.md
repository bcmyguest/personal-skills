# Forge commands

Read only your forge's section. Flags drift between CLI versions; confirm with `<cli> <subcommand> --help` when a call fails.

## GitHub (`gh`)

"Could not resolve to a Repository" on a repo that exists means the active `gh` account lacks access. Check `gh auth status`; run with another logged-in account for one shell via `export GH_TOKEN=$(gh auth token --user <login>)`.

- The PR: `gh pr view [<number> | <url>] --json number,url,state,baseRefName,headRefName,mergeable,reviewDecision`; no argument means the current branch. Check out its head: `gh pr checkout <number>`
- Top-level comments and review summaries: `gh pr view --json comments,reviews`
- Inline threads with resolved state (`gh pr view` omits it; use GraphQL):

  ```bash
  gh api graphql -F owner=<owner> -F repo=<repo> -F pr=<number> -f query='
  query($owner:String!,$repo:String!,$pr:Int!){repository(owner:$owner,name:$repo){
    pullRequest(number:$pr){reviewThreads(first:100){nodes{
      id isResolved isOutdated path line
      comments(first:50){nodes{databaseId author{login} body url}}}}}}}'
  ```

- Reply in a thread: `gh api repos/<owner>/<repo>/pulls/<number>/comments/<databaseId of the thread's first comment>/replies -f body='...'`
- Resolve a thread: `gh api graphql -F id=<thread id> -f query='mutation($id:ID!){resolveReviewThread(input:{threadId:$id}){thread{isResolved}}}'`
- Reply to a top-level comment: `gh pr comment <number> --body '...'`
- Jobs: `gh pr checks <number>` (`--watch` blocks until they finish); failing logs: `gh run view <run-id> --log-failed`; rerun: `gh run rerun <run-id> --failed`
- Children: `gh pr list --base <head-branch> --json number,headRefName`
- Parent: `gh pr list --head <base-branch> --json number,baseRefName`

## GitLab (`glab`)

`glab api` expands `:id` to the current project.

- The MR: `glab mr view [<iid> | <branch>] --output json` (fields `iid`, `state`, `target_branch`, `source_branch`, `has_conflicts`); no argument means the current branch. From a link, the `iid` is the number after `/-/merge_requests/`. Check out its head: `glab mr checkout <iid>`
- All discussions: `glab api "projects/:id/merge_requests/<iid>/discussions?per_page=100"`. An open thread has a note with `resolvable: true` and `resolved: false`; notes with `resolvable: false` and `system: false` are top-level comments.
- Reply in a thread: `glab api -X POST "projects/:id/merge_requests/<iid>/discussions/<discussion-id>/notes" -f body='...'`
- Resolve a thread: `glab api -X PUT "projects/:id/merge_requests/<iid>/discussions/<discussion-id>?resolved=true"`
- Top-level comment: `glab mr note <iid> --message '...'`
- Jobs: `glab ci status`; failing job log: `glab ci trace <job-id>`; rerun: `glab ci retry <job-id>`
- Children: `glab mr list --target-branch <source-branch>`
- Parent: `glab mr list --source-branch <target-branch>`
- A push to a new branch may auto-create an MR; check before opening one.

## Azure DevOps (`az repos`)

Needs the extension: `az extension add --name azure-devops`. Organization and project come from `az devops configure --defaults` or `--org` / `--project`.

- The PR: `az repos pr show --id <id>`, or for the current branch `az repos pr list --source-branch <branch> --status active`. From a link, the id is the number after `/pullrequest/`. Check out its head: `az repos pr checkout --id <id>`
- Threads (no first-class command):

  ```bash
  az devops invoke --area git --resource pullRequestThreads --api-version 7.1 \
    --route-parameters project=<project> repositoryId=<repo> pullRequestId=<id>
  ```

  An open thread has `status: "active"`; skip threads whose comments are all `commentType: "system"`. `threadContext.filePath` and `rightFileStart.line` locate inline threads.

- Reply in a thread: write `{"content": "...", "parentCommentId": <first comment id>, "commentType": 1}` to a file, then

  ```bash
  az devops invoke --area git --resource pullRequestThreadComments --api-version 7.1 \
    --http-method POST --in-file <file> \
    --route-parameters project=<project> repositoryId=<repo> pullRequestId=<id> threadId=<thread-id>
  ```

- Resolve a thread: the same `pullRequestThreads` call with `--http-method PATCH`, `threadId=<thread-id>` added to the route parameters, and `--in-file` holding `{"status": "fixed"}`.
- Jobs: `az repos pr policy list --id <id>`; build detail: `az pipelines runs show --id <build-id>`; rerun: `az repos pr policy queue --id <id> --evaluation-id <evaluation-id>`
- Children: `az repos pr list --target-branch <source-branch> --status active`
- Parent: `az repos pr list --source-branch <target-branch> --status active`
