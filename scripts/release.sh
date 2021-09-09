#!/bin/bash

REPO='https://github.com/tinkoff-ai/etna-ts'

GREEN=`tput setaf 2`
YELLOW=`tput setaf 3`
BOLD=`tput bold`
RESET=`tput sgr0`

PACKAGE_NAME=$(poetry version | cut -d ' ' -f1)

gh auth status || echo -e "\nPlease, auth with command: \n${BOLD}gh auth login --web${RESET}"

UPDATE_RULE=${1:-''}
if [[ $UPDATE_RULE != "minor" && $UPDATE_RULE != "patch" ]]; then
    echo -e "Please, specify update rule — 'minor' or 'patch' \nExample usage: bash release.sh minor"
    exit 1
fi

read -p "${BOLD}Do you really want to release $PACKAGE_NAME?${RESET} ${YELLOW}(y/д)${RESET} " -n 1 -r
echo    # (optional) move to a new line
if [[ ! $REPLY =~ ^[YyДд]$ ]]
then
    [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
fi

poetry version $UPDATE_RULE

version=$(poetry version --short)
message=":bomb: release $version"

git checkout -b release/$version
git commit -am "$message"
git push -u origin release/$version

gh pr create --title "$message" --body "# Great!
Please visit $REPO/releases/edit/$version to describe **release notes!**

Also you can find publishing task here $REPO/actions/workflows/publish.yml"

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
gh release create "$version" --title "$message" --notes "In progress..." --target $CURRENT_BRANCH
gh pr view --web

echo -e "\n${GREEN}${BOLD}Done!${RESET}"

