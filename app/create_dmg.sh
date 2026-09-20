#!/bin/bash
# Future DMGs always use the project artwork. Does not rebuild or modify the app.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="${1:?Usage: create_dmg.sh app-path output-dmg}"
OUTPUT="${2:?Output DMG path required}"
BACKGROUND="$ROOT/icons/DMG_Back.png"
[ -f "$BACKGROUND" ] || { echo "Missing background: $BACKGROUND" >&2; exit 1; }
codesign --verify --deep --strict "$APP"
WORK="$(mktemp -d -t yue-dmg)"
MOUNT="$WORK/mount"
cleanup() {
  if mount | grep -Fq " on $MOUNT "; then hdiutil detach "$MOUNT" -quiet || true; fi
  rm -rf "$WORK"
}
trap cleanup EXIT
mkdir -p "$WORK/staging/.background" "$MOUNT"
ditto "$APP" "$WORK/staging/YuE Studio.app"
sips -z 512 768 "$BACKGROUND" --out "$WORK/staging/.background/DMG_Back.png" >/dev/null
ln -s /Applications "$WORK/staging/Applications"
hdiutil create -srcfolder "$WORK/staging" -volname 'YuE Studio' -format UDRW "$WORK/editable.dmg" -quiet
hdiutil attach -nobrowse -mountpoint "$MOUNT" "$WORK/editable.dmg" -quiet
# Finder stores the background reference and icon positions in .DS_Store.
osascript - "$MOUNT" <<'APPLESCRIPT'
on run argv
    set diskFolder to POSIX file (item 1 of argv) as alias
    tell application "Finder"
        open diskFolder
        set diskWindow to container window of diskFolder
        set current view of diskWindow to icon view
        set toolbar visible of diskWindow to false
        set statusbar visible of diskWindow to false
        set bounds of diskWindow to {100, 100, 868, 612}
        set options to icon view options of diskWindow
        set arrangement of options to not arranged
        set icon size of options to 96
        set background picture of options to file ".background:DMG_Back.png" of diskFolder
        set position of item "YuE Studio.app" of diskFolder to {210, 280}
        set position of item "Applications" of diskFolder to {558, 280}
        update diskFolder without registering applications
        delay 2
        close diskWindow
    end tell
end run
APPLESCRIPT
hdiutil detach "$MOUNT" -quiet
hdiutil convert "$WORK/editable.dmg" -format UDZO -o "$OUTPUT" -ov -quiet
