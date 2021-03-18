while read pkg; do
    echo $pkg
    jq --arg pkg "$pkg" '.PACKAGE |= $pkg' config.json > config_tmp.json
    cat config_tmp.json > config.json
    echo -ne '\n' | java -jar ./paladin.jar
done < 'package.list'