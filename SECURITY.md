# Security Policy

## What this project is

dvision is a drone **simulator** and a set of tools around it. It is research
and development software: it does not fly aircraft, it is not certified for
any safety-related use, and nothing in it should be treated as flight-worthy.
Please keep that in mind when judging the impact of a finding.

## Supported versions

Development happens on `master`, and only `master` receives fixes. There are
no maintained release branches. Tagged versions are snapshots, not supported
lines.

| Version  | Supported |
| -------- | --------- |
| `master` | yes       |
| anything else | no   |

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Two private routes, either is fine:

1. **GitHub private vulnerability reporting** — on the repository's *Security*
   tab, "Report a vulnerability". This is the preferred route: it keeps the
   discussion, the fix and the advisory in one place.
2. **Email** — <dvision@wheresjames.com>.

Useful things to include, as far as you have them: what you were running, the
commit, what you expected, what happened, and the smallest reproduction you
can manage. A proof of concept is welcome but not required.

### What to expect

This is a small project maintained by one person, so these are honest
intentions rather than a contractual SLA:

* an acknowledgement within about a week;
* an assessment of whether it is a real issue, and its severity, after that;
* a fix on `master` and a published advisory crediting you, unless you would
  rather not be named.

Please give a reasonable window before disclosing publicly. If you have not
heard back in two weeks, assume the mail went astray and try the other route.

## Scope

**In scope** — anything that lets untrusted input reach code execution, or
that leaks data across a boundary the design says it should not cross:

* the shared-memory transports (`pymembus` channels, sensor rings, the command
  and status planes) — in particular any way a malformed record or an
  unexpected channel name causes memory corruption or arbitrary execution in a
  reader;
* map, profile and tour parsing — these read files and JSON from disk and are
  the most likely place for a crash on hostile input;
* the process launchers, especially `apps/dfgb`, which starts FlightGear and
  ffmpeg under a dedicated Xvfb display and builds command lines from
  configuration;
* dependency vulnerabilities that are actually reachable from this code.

**Out of scope** — known and accepted properties of a local development tool:

* the IPC transports are **unauthenticated by design**. Any process running as
  the same user can open a channel by name, read sensor data and issue
  commands. This is not a vulnerability report; it is the model. If you find a
  way to reach those channels from *another* user or from off the machine,
  that very much is one.
* denial of service by giving a component absurd input (a map that does not
  fit in memory, a profile with a million sensors);
* anything requiring an attacker who already has the ability to run code as the
  user;
* findings from an automated scanner with no demonstrated impact.

## What runs automatically

CI runs `pip-audit` against the pinned dependencies and `gitleaks` over the
full history on every push and nightly, and CodeQL on pushes to `master` and
weekly. These catch the ordinary cases; they are not a substitute for a report.
