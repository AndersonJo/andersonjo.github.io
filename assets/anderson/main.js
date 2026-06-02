const MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function andersonDate(input) {
    const date = new Date(input);
    if (isNaN(date.getTime())) return '';
    return MONTH_ABBR[date.getMonth()] + ' ' + date.getDate() + ' ' + date.getFullYear();
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function renderPosts(posts, searchText) {
    const list = document.getElementById('post-list');
    if (!list) return;

    const query = (searchText || '').toLowerCase();
    const filtered = query
        ? posts.filter(p =>
            String(p.title).toLowerCase().includes(query) ||
            (p.categories || []).some(c => c.toLowerCase().includes(query)) ||
            (p.tags || []).some(t => String(t).toLowerCase().includes(query))
          )
        : posts;

    list.innerHTML = filtered.map(function(post) {
        const tagTitle = escapeHtml((post.tags || []).join(', '));
        const title = escapeHtml(post.title || '');
        const url = escapeHtml(post.url || '#');
        const cats = (post.categories || []).join(' ');
        return '<li>'
            + '<span class="post-date">' + andersonDate(post.date) + '</span> - '
            + '<span class="post-category">' + escapeHtml(cats) + '</span>'
            + '<a class="post-comment-count" href="' + url + '#disqus_thread"></a>'
            + '<div><a class="post-link post-tag" href="' + url + '" title="' + tagTitle + '">'
            + title
            + '</a></div>'
            + '</li>';
    }).join('');

    if (typeof bootstrap !== 'undefined') {
        list.querySelectorAll('[title]').forEach(function(el) {
            new bootstrap.Tooltip(el);
        });
    }
}

function initPostSearch(posts) {
    renderPosts(posts, '');
    var searchInput = document.getElementById('search-input');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            renderPosts(posts, this.value);
        });
    }
}

function loadBibleVerse() {
    fetch('/assets/anderson/bible.csv')
        .then(function(r) { return r.text(); })
        .then(function(text) {
            var lines = text.split('\n').filter(function(l) { return l.trim().length > 0; });
            if (!lines.length) return;
            var el = document.getElementById('bible-statement');
            if (el) el.textContent = lines[Math.floor(Math.random() * lines.length)];
        })
        .catch(function() {});
}

function loadFastCategories(posts) {
    fetch('/assets/anderson/fast_categories.csv')
        .then(function(r) { return r.text(); })
        .then(function(text) {
            var cats = text.split('\n').filter(function(l) { return l.trim().length > 0; });
            var container = document.getElementById('fast-categories');
            if (!container || !cats.length) return;
            container.innerHTML = cats.map(function(cat) {
                return '<button type="button" class="fast_category" data-cat="' + escapeHtml(cat) + '">' + escapeHtml(cat) + '</button>';
            }).join('');
            container.querySelectorAll('.fast_category').forEach(function(btn) {
                btn.addEventListener('click', function() {
                    var cat = this.dataset.cat;
                    var input = document.getElementById('search-input');
                    if (input) { input.value = cat; }
                    renderPosts(posts, cat);
                });
            });
        })
        .catch(function() {});
}
