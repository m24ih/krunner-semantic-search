outdir ?= .

srpm:
	mkdir -p $(outdir)
	git archive --format=tar.gz --prefix=krunner-semantic-search-0.1.0/ -o krunner-semantic-search-0.1.0.tar.gz HEAD 2>/dev/null || true
	spectool -g -R packaging/fedora/krunner-semantic-search.spec || true
	rpmbuild -bs --define "_sourcedir $(shell pwd)" --define "_srcrpmdir $(outdir)" packaging/fedora/krunner-semantic-search.spec

.PHONY: srpm
