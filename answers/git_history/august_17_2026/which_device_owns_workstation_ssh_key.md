# Which device owns the GitHub SSH key (2026-08-17)

The key used by `git push` in this Cursor session is on the **Pop!_OS workstation**, not the MacBook.

| Fact | Value |
|------|-------|
| This session’s host | `pop-os` (Linux 6.17.9, user files under `/home/alex`) |
| Key path | `/home/alex/.ssh/id_ed25519` (+ `.pub`) |
| Created | 2025-10-02 15:39 |
| Comment | `aromanus@gmail.com` (email, not a hostname) |
| Fingerprint | `SHA256:OEIUVU7IVubCI4pU4E6zAPLuAREBOIxMHNaw2CUgHY0` |
| Other keys in that `~/.ssh` | none |

Cursor is SSH’d from the MacBook into this host, so git runs on the workstation and presents this key. The MacBook’s own `~/.ssh` is not involved unless a push is done locally on the laptop.
