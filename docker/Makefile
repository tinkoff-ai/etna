COMMIT=$(shell  git rev-parse --short HEAD)
TAG=$(COMMIT)

image_build:
	docker build -f ${DOCKERFILE_PATH} -t ${IMAGE_NAME}:${TAG} .
