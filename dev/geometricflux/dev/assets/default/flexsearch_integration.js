// custom search widget
(function() {
    const MAX_RESULTS = 40
    const flexsearchIdx = new FlexSearch.Document({
        document: {
            id: 'id',
            store: ['title', 'pagetitle', 'ref'],
            index: [
                {
                    field: 'content',
                }
            ]
        },
    });

    let successfullyLoadedIndex = null

    function loadIndex(flexsearchIdx) {
        const input = document.getElementById('search-input')
        input.setAttribute('placeholder', 'Loading...')
        successfullyLoadedIndex = false
        const keys = ['content.cfg', 'content.ctx', 'content.map', 'reg', 'store']
        const rootPath = window.MULTIDOCUMENTER_ROOT_PATH ?? '/'
        const promises = keys.map(key => {
            return new Promise((resolve, reject) => {
                fetch(`${rootPath}search-data/${key}.json`).then(r => {
                    if (r && r.ok) {
                        r.json().then(idx => {
                            flexsearchIdx.import(key, idx)
                            resolve()
                        }).catch(() => {
                            reject()
                        })
                    } else {
                        reject()
                    }
                }).catch(() => {
                    reject()
                })
            })
        })

        Promise.all(promises).then(() => {
            input.setAttribute('placeholder', 'Search...')
            successfullyLoadedIndex = true
        }).catch(() => {
            input.setAttribute('placeholder', 'Error loading search data...')
            successfullyLoadedIndex = false
        })
    }

    function registerSearchListener() {
        const input = document.getElementById('search-input')
        const suggestions = document.getElementById('search-result-container')

        let lastQuery = ''

        function runSearch() {
            if (successfullyLoadedIndex === null) {
                loadIndex(flexsearchIdx)
            } else if (successfullyLoadedIndex === false) {
                return
            }
            const query = input.value

            if (flexsearchIdx && query !== lastQuery) {
                lastQuery = query

                console.time('search')
                let results = flexsearchIdx.search(query, {
                    limit: MAX_RESULTS,
                    enrich: true
                })
                console.timeEnd('search')

                if (results.length > 0) {
                    buildResults(results[0].result.map(r => r.doc))
                } else {
                    suggestions.classList.add('hidden')
                }
            }
        }

        input.addEventListener('keyup', ev => {
            runSearch()
        })

        input.addEventListener('keydown', ev => {
            if (ev.key === 'ArrowDown') {
                suggestions.firstChild.firstChild.focus()
                ev.preventDefault()
                return
            } else if (ev.key === 'ArrowUp') {
                suggestions.lastChild.firstChild.focus()
                ev.preventDefault()
                return
            }
        })

        suggestions.addEventListener('keydown', ev => {
            if (ev.target.dataset.index !== undefined) {
                const li = ev.target.parentElement
                if (ev.key === 'ArrowDown') {
                    const el = li.nextSibling
                    if (el) {
                        el.firstChild.focus()
                        ev.preventDefault()
                    } else {
                        input.focus()
                    }
                } else if (ev.key === 'ArrowUp') {
                    const el = li.previousSibling
                    if (el) {
                        el.firstChild.focus()
                        ev.preventDefault()
                    } else {
                        input.focus()
                    }
                }
            }
        })

        input.addEventListener('focus', ev => {
            runSearch()
        })
    }

    function buildResults(results) {
        const suggestions = document.getElementById('search-result-container')

        suggestions.classList.remove('hidden')

        console.log(results)

        const children = results.slice(0, MAX_RESULTS - 1).map((r, i) => {
            const entry = document.createElement('li')
            entry.classList.add('suggestion')
            const link = document.createElement('a')
            link.setAttribute('href', r.ref)
            link.dataset.index = i
            const page = document.createElement('span')
            page.classList.add('page-title')
            page.innerText = r.pagetitle
            const section = document.createElement('span')
            section.innerText = ' > ' + r.title
            section.classList.add('section-title')
            link.appendChild(page)
            link.appendChild(section)
            entry.appendChild(link)
            return entry
        })
        suggestions.replaceChildren(
            ...children
        )
    }

    function initialize() {
        registerSearchListener()

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
