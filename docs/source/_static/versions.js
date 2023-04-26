$(document).ready(function () {
    $.getJSON('/versions.json', function (data) {
        if (data.length > 0) {
            var versions = [];
            $.each(data, function (index, value) {
                versions.push(value);
            });
            var dl = document.getElementById('docs-versions');
            $.each(versions, function (i, v) {
                var version = versions[i];
                dl.innerHTML = dl.innerHTML +
                    '<dd><a href="/etna/' + version + '">' + version + '</a></dd>'
            });

        }
    });
});