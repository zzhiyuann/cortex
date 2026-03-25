# VHS Demo Tape for Cortex
# Records: help, status dashboard, info overview
# Run: vhs cortex-demo.tape

Output cortex-demo.gif

Set Shell zsh
Set FontSize 14
Set Width 1100
Set Height 650
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms
Set WindowBar Colorful
Set BorderRadius 8
Set WindowBarSize 40
Set CursorBlink false

# Install
Type 'echo "$ pip install cortex-cli-agent"'
Enter
Sleep 2s

# Show CLI help
Type "cortex --help"
Enter
Sleep 4s

# Show status dashboard
Type "cortex status"
Enter
Sleep 4s

# Show ecosystem overview
Type "cortex info"
Enter
Sleep 4s

# Show doctor command
Type "cortex doctor"
Enter
Sleep 4s
