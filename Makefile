.PHONY: doctor train bench test serve demo

doctor:
	python -m engramforge doctor

train:
	python -m engramforge train

bench:
	python -m engramforge bench

test:
	python -m unittest discover -s tests -t .

serve:
	python -m engramforge serve

demo:
	python -m engramforge demo
