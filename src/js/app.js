/* sweetScroll for in-page anchor links */
document.addEventListener('DOMContentLoaded', function () {
  new SweetScroll({});
}, false);

// #region agent log
(function () {
  var ENDPOINT =
    'http://127.0.0.1:7541/ingest/76d61dc9-79cb-4ec9-b447-afdcea5ca461';
  function dbg(payload) {
    payload.sessionId = '622099';
    payload.timestamp = Date.now();
    fetch(ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Debug-Session-Id': '622099',
      },
      body: JSON.stringify(payload),
    }).catch(function () {});
  }
  document.addEventListener('DOMContentLoaded', function () {
    var cssLink = document.querySelector('link[href*="main.css"]');
    var cssHref = cssLink ? cssLink.href : '';
    var resolvedFontUrl = '';
    try {
      if (cssHref) {
        resolvedFontUrl = new URL(
          '../fonts/fontawesome-webfont.woff2?v=4.7.0',
          cssHref
        ).href;
      }
    } catch (e) {
      resolvedFontUrl = 'resolve-error:' + String(e.message);
    }
    var icon = document.querySelector('.header-icons .fa');
    var iconFF = icon ? window.getComputedStyle(icon).fontFamily : '(no .header-icons .fa)';
    var beforeFF = '';
    var beforeContent = '';
    if (icon) {
      try {
        beforeFF = window.getComputedStyle(icon, '::before').fontFamily;
        beforeContent = window.getComputedStyle(icon, '::before').content;
      } catch (e) {
        beforeFF = 'pseudo-error';
      }
    }
    dbg({
      location: 'app.js:icon-styles',
      message: 'computed styles for header FA icon',
      data: {
        cssHref: cssHref,
        resolvedFontUrl: resolvedFontUrl,
        iconFontFamily: iconFF,
        beforeFontFamily: beforeFF,
        beforeContent: beforeContent,
      },
      hypothesisId: 'H3',
      runId: 'post-font-path-fix',
    });
    if (resolvedFontUrl && resolvedFontUrl.indexOf('error') === -1) {
      fetch(resolvedFontUrl, { method: 'HEAD', cache: 'no-store' })
        .then(function (r) {
          dbg({
            location: 'app.js:font-head',
            message: 'HEAD font file',
            data: {
              url: resolvedFontUrl,
              status: r.status,
              ok: r.ok,
              ct: r.headers.get('content-type'),
            },
            hypothesisId: 'H1',
            runId: 'post-font-path-fix',
          });
        })
        .catch(function (err) {
          dbg({
            location: 'app.js:font-head',
            message: 'HEAD font failed',
            data: { url: resolvedFontUrl, err: String(err) },
            hypothesisId: 'H1',
            runId: 'post-font-path-fix',
          });
        });
    }
    if (document.fonts && document.fonts.check) {
      dbg({
        location: 'app.js:fonts-check',
        message: 'document.fonts.check FontAwesome',
        data: {
          unquoted: document.fonts.check('1em FontAwesome'),
          quoted: document.fonts.check('1em "FontAwesome"'),
        },
        hypothesisId: 'H2',
        runId: 'post-font-path-fix',
      });
    }
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(function () {
        dbg({
          location: 'app.js:fonts-ready',
          message: 'after document.fonts.ready',
          data: {
            unquoted: document.fonts.check('1em FontAwesome'),
            quoted: document.fonts.check('1em "FontAwesome"'),
          },
          hypothesisId: 'H2',
          runId: 'post-font-path-fix',
        });
      });
    }
  });
})();
// #endregion

document.addEventListener(
  'click',
  function (e) {
    var btn = e.target.closest('.js-email-reveal');
    if (!btn) return;
    var b64 = btn.getAttribute('data-email-b64');
    if (!b64) return;
    e.preventDefault();
    var wrap = btn.closest('.email-reveal-wrap');
    if (!wrap) return;
    try {
      var addr = atob(b64);
      var a = document.createElement('a');
      a.href = 'mailto:' + addr;
      a.textContent = addr;
      a.className = 'email-reveal-link';
      wrap.replaceChild(a, btn);
    } catch (err) {
      /* ignore invalid base64 */
    }
  },
  false
);
