// custom search widget
(function() {
    function initialize() {
        stork.register('multidocumenter', '/stork.st')

        document.body.addEventListener('keydown', ev => {
            if (document.activeElement === document.body && (ev.key === '/' || ev.key === 's')) {
                document.getElementById('search-input').focus()
                ev.preventDefault()
            }
        })
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize)
    } else {
        initialize()
    };
})()
