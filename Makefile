web:
	cd docs;\
	make html;\
	cd ../;\
	git add docs/build/html;\
	git commit -m "Update Website";\
	git subtree push --prefix docs/build/html/ origin gh-pages;\
