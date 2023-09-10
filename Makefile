all: 
	@for a in $$(ls); do \
		if [ -d $$a -a "$$a" != "Common" ]; then \
			echo "compiling $$a"; \
			$(MAKE) -C $$a; \
		fi; \
	done;
	@echo "Completed..."


clean:
	@for a in $$(ls); do \
		if [ -d $$a -a "$$a" != "Common" ]; then \
			echo "Cleaning $$a"; \
			$(MAKE) -C $$a clean; \
		fi; \
	done;
	@echo "Cleaning completed..."
