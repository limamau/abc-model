# Docs

To generate the API reference, run the following command inside the `docs` directory.

```
rm -rf source
sphinx-apidoc -f -e -t _templates -o source/api ../src
make clean html
```
