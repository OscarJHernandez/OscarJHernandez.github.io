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
    var list = document.querySelectorAll('.header-icons svg.header-icon');
    var first = list[0];
    var r = first ? first.getBoundingClientRect() : null;
    dbg({
      location: 'app.js:header-svgs',
      message: 'header inline SVG icons (post webfont bypass)',
      data: {
        svgCount: list.length,
        firstWidthPx: r ? Math.round(r.width * 100) / 100 : null,
        firstHeightPx: r ? Math.round(r.height * 100) / 100 : null,
      },
      hypothesisId: 'FIX-SVG',
      runId: 'svg-icon-fix',
    });
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
