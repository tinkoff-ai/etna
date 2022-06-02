How to create new release?
==========================

**First of all you should be collaborator of ETNA repository.**

Releasing new version of ETNA is really simple:
    1. Change CHANGELOG.md file:
        1.1. Collect all changes and delete empty bullets.

        1.2. Specify version and date of the release (*Note: we use semantic versioning*)
    2. Create pull request with the changes above.
    3. Wait until the PR is approved by any team member.
    4. After PR approved and merged into the master branch create new release_ and tag this release.
    5. That's all! Our CI/CD will take care of everything else.

.. _release: https://github.com/tinkoff-ai/etna/releases