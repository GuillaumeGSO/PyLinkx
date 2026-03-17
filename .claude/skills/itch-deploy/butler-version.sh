#!/bin/bash
# Print the current live version number for the html5 channel
butler status GuillaumeGSO/pygame-linkx-test:html5 2>&1 | grep '| html5' | awk -F'|' '{print $5}' | tr -d ' '
