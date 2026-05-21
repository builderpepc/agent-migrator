---
name: cli-explorer
description: Use cli-explorer whenever you need to interact with a running terminal application that requires keyboard input, visual inspection, or step-by-step navigation — things that go beyond a single Bash command. This includes: exploring an unfamiliar CLI tool interactively, driving a TUI app (vim, htop, ncurses, tmux, fzf, interactive REPL), testing a command-line application end-to-end, sending keystrokes to a running process (ctrl+c, escape, arrow keys), waiting for specific output before continuing, scrolling through terminal history, capturing PNG screenshots of terminal output, saving text snapshots for regression testing, and recording GIF/video demos. Use this instead of Bash when the program needs back-and-forth interaction, not just a one-shot command. Trigger this skill whenever the user asks to "explore", "try out", "interact with", or "test" a CLI or TUI application.
allowed-tools: Bash(cli-explorer:*)
---

# cli-explorer

`cli-explorer` runs any program in a real PTY (pseudo-terminal) session that you can fully control: send text, press keys, scroll, wait for output, take screenshots, and save text snapshots.

**Key behaviors:**
- Every `write`, `key`, `scroll`, `wait`, and `resize` command automatically prints the current screen — no need for a separate `snapshot` call.
- The daemon starts automatically on first use and shuts down 30s after the last session closes.
- Use `-s <name>` (e.g. `-s editor`) to run multiple concurrent sessions.
- **Windows:** use `powershell.exe` or `cmd.exe` as the shell command; on Unix use `bash` or `zsh`.

## Quick Start

```bash
cli-explorer open bash                          # start a shell session
cli-explorer wait "$ "                          # wait for shell prompt
cli-explorer write "ls -la"                     # type text
cli-explorer key enter                          # press Enter
cli-explorer snapshot                           # print current screen
cli-explorer screenshot --output /tmp/screen.png  # save PNG
cli-explorer close                              # end session
```

## Command Reference

### Starting / stopping

```bash
cli-explorer open <command> [args...]           # launch a program in a PTY
  --cols <n>           # width (default 80)
  --rows <n>           # height (default 24)
  --cwd <path>         # working directory
  --snapshot-dir <dir> # where to save snapshot files

cli-explorer close                              # close default session
cli-explorer close --all                        # close every session
cli-explorer kill-all                           # force-kill daemon + all sessions
cli-explorer list                               # list active sessions
```

### Input

```bash
cli-explorer write <text>     # type text into the terminal
cli-explorer key <keyname>    # press a key
```

Key names: `enter`, `tab`, `backspace`, `escape`, `up`, `down`, `left`, `right`,
`ctrl+c`, `ctrl+d`, `ctrl+z`, `ctrl+a`, `ctrl+e`, `ctrl+k`, `ctrl+l`, `ctrl+n`,
`ctrl+p`, `ctrl+r`, `ctrl+u`, `ctrl+w`, `pageup`, `pagedown`, `home`, `end`,
`delete`, `insert`, `f1`–`f12`

### Waiting

```bash
cli-explorer wait <text>                # block until text appears on screen
  --regex              # treat <text> as a regex
  --timeout <ms>       # timeout in ms (default 5000)
```

### Viewing the screen

```bash
cli-explorer snapshot                   # current viewport as plain text
cli-explorer snapshot --full            # entire scrollback buffer
cli-explorer cursor                     # print cursor position {"x":N,"y":N}
```

### Scrolling

```bash
cli-explorer scroll up|down|top|bottom  # scroll the viewport
  --lines <n>          # lines to scroll for up/down (default 3)
cli-explorer scroll-to-line <n>         # jump to absolute buffer line (0-indexed)
```

### Text snapshot regression testing

```bash
cli-explorer snapshot save <name>       # save current screen to <name>.txt
cli-explorer snapshot assert <name>     # assert screen matches saved snapshot (exit 1 + diff on fail)
cli-explorer snapshot update <name>     # overwrite saved snapshot with current screen
# All three accept --full for the scrollback buffer
```

### Image (PNG) snapshot regression testing

```bash
cli-explorer screenshot                           # capture PNG → base64 JSON
cli-explorer screenshot --output <path>           # capture PNG → save to file
cli-explorer screenshot assert <name>             # pixel-level comparison (exit 1 on fail)
  --pixel-threshold <0.0-1.0>  # fraction of pixels allowed to differ
  --max-diff-pixels <n>        # absolute pixel count allowed to differ
cli-explorer screenshot update <name>             # update saved PNG baseline
```

### Session targeting

```bash
cli-explorer -s <name> open vim myfile.txt  # open named session
cli-explorer -s <name> write "ihello"       # target named session
# Environment: CLI_EXPLORER_SESSION=<name> also works
```

### Recording

```bash
cli-explorer record start --output /tmp/demo.gif   # start recording (.gif/.mp4/.webm)
cli-explorer record stop                            # stop + save
```

### Skill installation

```bash
cli-explorer install --skills              # project-local: ./.claude/skills/cli-explorer/
cli-explorer install --skills --global     # user-global:   ~/.claude/skills/cli-explorer/
cli-explorer install --skills --dir <path> # explicit directory
```

### Output format

```
### Session: default

### Snapshot
<screen content here>
```

Add `--raw` to any command to suppress the headers and get just the screen content.

## Workflow Examples

### Interactively explore a shell

```bash
cli-explorer open bash
cli-explorer wait "$ "
cli-explorer write "cat /etc/os-release"
cli-explorer key enter
cli-explorer wait "$ "
cli-explorer snapshot
```

### Drive vim end-to-end

```bash
cli-explorer open vim notes.txt
cli-explorer wait "notes.txt"
cli-explorer write "iHello from the agent"   # 'i' = insert mode
cli-explorer key escape
cli-explorer write ":wq"
cli-explorer key enter
```

### Test an interactive Python REPL

```bash
cli-explorer open python3
cli-explorer wait ">>>"
cli-explorer write "print(2 + 2)"
cli-explorer key enter
cli-explorer wait ">>>"
cli-explorer snapshot
```

### Regression test a TUI

```bash
cli-explorer open myapp --snapshot-dir ./snapshots
cli-explorer wait "Ready"
cli-explorer snapshot save main-screen       # create baseline on first run
cli-explorer snapshot assert main-screen     # verify on subsequent runs
cli-explorer screenshot assert main-screen   # also check visually
```

### Capture a TUI screenshot

```bash
cli-explorer open htop
cli-explorer wait "CPU"
cli-explorer screenshot --output /tmp/htop.png
cli-explorer key q
```
