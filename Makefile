XPECTRASS_DOCS := docs

.PHONY: docs clean

docs:
	cd $(XPECTRASS_DOCS) && sphinx-build -b html . _build/html

clean:
	rm -rf $(XPECTRASS_DOCS)/_build

livehtml:
	cd $(XPECTRASS_DOCS) && sphinx-autobuild . _build/html
