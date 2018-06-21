SHELL:=/bin/bash
THIS:=$(lastword $(MAKEFILE_LIST))
PREFIX=/opt/t3ps-1.0
LINKPREFIX=

.PHONY: install uninstall

install:
	@echo -n "Please enter installation directory: [$(PREFIX)] "; \
    read PREFIX; \
    eval PREFIX=`echo $$PREFIX`; \
    if [ ! "$$PREFIX" ]; \
    then \
        PREFIX=$(PREFIX); \
    fi; \
    echo "Installing to" $$PREFIX; \
    if [ -w "$$PREFIX" ] || [ -w "`dirname $$PREFIX`" ]; \
    then \
        mkdir -vp "$$PREFIX" && \
        cp -v src/t3ps "$$PREFIX/t3ps" && \
        mkdir -vp "$$PREFIX/processors" && \
        cp -v src/processors/ExampleProcessor.py "$$PREFIX/processors" && \
        cp -v src/processors/SimpleProcessor.py "$$PREFIX/processors" && \
        cp -v src/processors/SLHAProcessor.py "$$PREFIX/processors" && \
        cp -v src/processors/ProcessorChain.py "$$PREFIX/processors" \
        || exit 1; \
        sed -i "s:^PREFIX=.*:PREFIX=$$PREFIX:" "$(THIS)"; \
    else \
        echo "Installation directory not writable!"; \
        exit 1; \
    fi; \
    LINKPREFIX=$(LINKPREFIX); \
    if [ -w /usr/local/bin ]; \
    then \
        echo -n "Do you want to create a link in /usr/local/bin? [Y/n] "; \
        read ANSWER; \
        if [ "$$ANSWER" != "n" ]; \
        then \
            LINKPREFIX=/usr/local/bin; \
        fi; \
    elif [[ ":$$PATH:" == *":$$HOME/bin:"* || ":$$PATH:" == *":~/bin:"* ]]; then\
        echo -n "Do you want to create a link in $$HOME/bin? [Y/n] "; \
        read ANSWER; \
        if [ "$$ANSWER" != "n" ]; \
        then \
            LINKPREFIX="$$HOME/bin"; \
        fi; \
    fi; \
    if [ -n "$$LINKPREFIX" ]; then \
        ln -vfs "$$PREFIX/t3ps" "$$LINKPREFIX/t3ps"; \
        chmod -v u+x "$$LINKPREFIX/t3ps"; \
        chmod -v g+x "$$LINKPREFIX/t3ps"; \
        chmod -v o+x "$$LINKPREFIX/t3ps"; \
        sed -i "s:^LINKPREFIX=.*:LINKPREFIX=$$LINKPREFIX:" "$(THIS)"; \
    else \
        echo "No link to main executable created."; \
    fi; \
    cp -v $(THIS) "$$PREFIX/Makefile"

uninstall:
	rm --preserve-root "$(PREFIX)/t3ps"
	rm --preserve-root "$(PREFIX)/processors/ExampleProcessor.py" "$(PREFIX)/processors/ExampleProcessor.py"[oc]
	rm --preserve-root "$(PREFIX)/processors/SimpleProcessor.py" "$(PREFIX)/processors/SimpleProcessor.py"[oc]
	rm --preserve-root "$(PREFIX)/processors/SLHAProcessor.py" "$(PREFIX)/processors/SLHAProcessor.py"[oc]
	rm --preserve-root "$(PREFIX)/processors/ProcessorChain.py" "$(PREFIX)/processors/ProcessorChain.py"[oc]
ifdef LINKPREFIX
	rm --preserve-root "$(LINKPREFIX)/t3ps";
endif
	rmdir "$(PREFIX)/processors"
	rm --preserve-root "$(PREFIX)/Makefile"
	rmdir "$(PREFIX)"
