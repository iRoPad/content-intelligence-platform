.PHONY: install run test ask clean

install:
	pip install -r requirements.txt

run:
	python -m ciq run

test:
	python -m unittest discover -s tests -v

ask:
	python -m ciq ask "$(Q)"

clean:
	rm -rf warehouse data/raw
