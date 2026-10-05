# Maintainer Notes

Notes for maintainers of Scen-Opt. Users do not need any of this.

## Releasing a new version

Create a GitHub release with a tag of the form `vX.Y` (e.g. `v1.0`). Publishing the release:
- archives it on Zenodo with a new version DOI (the GitHub integration must be enabled for this repository at <https://zenodo.org/account/settings/github/>);
- builds and pushes the Docker image `ghcr.io/kiguli/scen-opt:vX.Y` (`.github/workflows/docker-publish.yml`).

Do not move or delete published tags: the paper and the Zenodo archive refer to them.

## Docker image

`.github/workflows/docker-publish.yml` builds the image on every push to `master` (tags `latest`, `master`, `sha-<commit>`) and on every `v*` tag. The package must stay **public** (package settings at <https://github.com/users/Kiguli/packages/container/package/scen-opt>) so that `docker pull` works without logging in.

To publish an image by hand:

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin   # token with write:packages
docker build -t ghcr.io/kiguli/scen-opt:latest .
docker push ghcr.io/kiguli/scen-opt:latest
```

## Web app deployment

`.github/workflows/deploy.yml` deploys every push to `master` to the server behind <https://scen-opt.woodingben.com>. The repository used to be called **Scen-O-Con**; the workflow still falls back to the old `scen-o-con` directory and systemd unit if the server has not been migrated. To migrate the server:

```bash
sudo systemctl stop scen-o-con && sudo systemctl disable scen-o-con
sudo mv /srv/scen-o-con.woodingben.com /srv/scen-opt.woodingben.com
sudo mv /etc/systemd/system/scen-o-con.service /etc/systemd/system/scen-opt.service
sudo systemctl daemon-reload && sudo systemctl enable --now scen-opt
```
