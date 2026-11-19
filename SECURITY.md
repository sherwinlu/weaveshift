# Security Policy

## Supported versions

Security fixes are applied to the latest release on the default branch.

| Version | Supported |
| ------- | --------- |
| 1.0.x   | Yes       |

## Reporting a vulnerability

Please do **not** open a public GitHub issue for security reports.

Email **sherwinlu@gmail.com** with:

- A description of the issue
- Steps to reproduce
- Impact assessment, if known

You should receive an acknowledgment within a few business days. We will coordinate disclosure and a fix before publishing details publicly when appropriate.

## Scope

WeaveShift is a local CLI audio tool. Reports are in scope when they describe:

- Remote code execution through malicious project inputs processed by WeaveShift
- Unsafe file handling that enables path traversal or arbitrary writes outside intended output paths
- Dependency or supply-chain issues that affect installed WeaveShift users

General audio quality, scoring heuristics, and DJ workflow preferences are not security issues.
