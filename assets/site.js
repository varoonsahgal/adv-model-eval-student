/* Eval Labs hub · site.js (no dependencies) */
(function () {
  "use strict";
  var root = document.documentElement;
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (/[?&]static\b/.test(location.search)) { root.classList.add("static"); reduce = true; }
  function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } }

  /* ---- theme toggle */
  var tb = document.querySelector(".theme-toggle");
  if (tb) tb.addEventListener("click", function () {
    var cur = root.getAttribute("data-theme") ||
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    var next = cur === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    store("theme", next);
  });

  /* ---- the all-green dashboard */
  var dash = document.getElementById("dash");
  if (dash) {
    var tiles = Array.prototype.slice.call(dash.querySelectorAll(".tile"));
    var msg = dash.querySelector(".dash-msg");
    var statusText = dash.querySelector(".status-text");
    tiles.forEach(function (t) {
      function toggle(e) {
        if (e.target.closest("a") && t.classList.contains("flipped")) return; // follow the link
        e.preventDefault();
        t.classList.toggle("flipped");
        var link = t.querySelector(".back a");
        if (link) link.tabIndex = t.classList.contains("flipped") ? 0 : -1;
      }
      t.addEventListener("click", toggle);
      t.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") toggle(e); });
    });
    var ship = dash.querySelector(".ship");
    if (ship) ship.addEventListener("click", function () {
      if (dash.classList.contains("busted")) {
        dash.classList.remove("busted");
        tiles.forEach(function (t) { t.classList.remove("flipped"); });
        statusText.textContent = "All checks passed";
        ship.textContent = "Ship v2 ✓";
        msg.textContent = "Hover or tap a tile. Each green number has a lab that breaks it.";
        return;
      }
      dash.classList.add("busted");
      if (!reduce) { dash.classList.remove("shake"); void dash.offsetWidth; dash.classList.add("shake"); }
      statusText.textContent = "Not so fast";
      ship.textContent = "Reset the dashboard";
      msg.textContent = "Every green tile is a question you haven’t asked yet. Each one is a lab.";
      tiles.forEach(function (t, i) {
        setTimeout(function () {
          t.classList.add("flipped");
          var link = t.querySelector(".back a"); if (link) link.tabIndex = 0;
        }, reduce ? 0 : 140 * i + 250);
      });
    });
  }

  /* ---- lab filters */
  var chips = document.querySelectorAll(".chip[data-filter]");
  var cards = document.querySelectorAll(".lab-card");
  var shown = document.getElementById("shown");
  chips.forEach(function (c) {
    c.addEventListener("click", function () {
      var f = c.getAttribute("data-filter");
      chips.forEach(function (x) { x.setAttribute("aria-pressed", x === c ? "true" : "false"); });
      var n = 0;
      cards.forEach(function (card) {
        var kind = card.getAttribute("data-kind"), fam = card.getAttribute("data-family");
        var ok = f === "all" || (f === "core" && kind === "core") || (f === "side" && kind !== "core") || fam === f;
        card.hidden = !ok;
        if (ok) { n++; card.classList.add("in"); }
      });
      if (shown) shown.textContent = n;
    });
  });

  /* ---- comic lightbox */
  var box = document.querySelector("dialog.lightbox");
  if (box && typeof box.showModal === "function") {
    var img = box.querySelector("img"), cap = box.querySelector(".cap");
    document.addEventListener("click", function (e) {
      var b = e.target.closest("button.comic, button.zoomable");
      if (!b) return;
      img.src = b.getAttribute("data-full");
      img.alt = (b.getAttribute("data-alt") || b.getAttribute("data-cap") || "");
      cap.textContent = b.getAttribute("data-cap") || "";
      box.showModal();
    });
    box.addEventListener("click", function (e) { if (e.target === box || e.target.closest("button")) box.close(); });
  }

  /* ---- pre-flight checklist (remembered in this browser only) */
  document.querySelectorAll("input[data-check]").forEach(function (cb) {
    var key = "check:" + cb.getAttribute("data-check");
    if (store(key) === "1") cb.checked = true;
    cb.addEventListener("change", function () { store(key, cb.checked ? "1" : "0"); });
  });

  /* ---- count-up stats */
  function countUp(el) {
    var end = parseInt(el.getAttribute("data-count"), 10);
    if (reduce || !end) { el.textContent = end; return; }
    var t0 = null;
    function step(t) {
      if (!t0) t0 = t;
      var p = Math.min(1, (t - t0) / 1100);
      el.textContent = Math.round(end * (1 - Math.pow(1 - p, 3)));
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  /* ---- reveal on scroll */
  var els = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window && !reduce) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); }
      });
    }, { rootMargin: "0px 0px -8% 0px" });
    els.forEach(function (el) { io.observe(el); });
    var counts = document.querySelectorAll("[data-count]");
    var io2 = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { countUp(en.target); io2.unobserve(en.target); } });
    });
    counts.forEach(function (el) { io2.observe(el); });
  } else {
    els.forEach(function (el) { el.classList.add("in"); });
  }

  /* ---- glossary search */
  var q = document.getElementById("q");
  if (q) {
    var rows = Array.prototype.slice.call(document.querySelectorAll(".g-row"));
    var heads = Array.prototype.slice.call(document.querySelectorAll(".g-letter"));
    var count = document.getElementById("g-count");
    var empty = document.querySelector(".g-empty");
    q.addEventListener("input", function () {
      var v = q.value.trim().toLowerCase(), n = 0;
      rows.forEach(function (r) { var ok = !v || r.getAttribute("data-text").indexOf(v) !== -1; r.hidden = !ok; if (ok) n++; });
      heads.forEach(function (h) {
        var el = h.nextElementSibling, any = false;
        while (el && !el.classList.contains("g-letter")) { if (!el.hidden) any = true; el = el.nextElementSibling; }
        h.hidden = !any;
      });
      count.textContent = n;
      if (empty) empty.hidden = n !== 0;
    });
  }

  /* ---- scroll progress */
  var sb = document.querySelector(".scrollbar");
  if (sb) {
    var onScroll = function () {
      var h = document.documentElement, max = h.scrollHeight - h.clientHeight;
      sb.style.width = (max > 0 ? Math.min(100, 100 * h.scrollTop / max) : 0) + "%";
    };
    window.addEventListener("scroll", onScroll, { passive: true }); onScroll();
  }

  /* ---- lab cards: tilt + cursor glow (pointer devices only) */
  var canTilt = window.matchMedia && window.matchMedia("(hover: hover)").matches && !reduce;
  document.querySelectorAll(".lab-card").forEach(function (card) {
    if (!card.querySelector(".glow")) { var g = document.createElement("span"); g.className = "glow"; card.appendChild(g); }
    if (!canTilt) return;
    card.addEventListener("pointermove", function (e) {
      var r = card.getBoundingClientRect();
      var px = (e.clientX - r.left) / r.width, py = (e.clientY - r.top) / r.height;
      card.style.setProperty("--mx", (px * 100).toFixed(1) + "%");
      card.style.setProperty("--my", (py * 100).toFixed(1) + "%");
      card.style.setProperty("--ry", ((px - .5) * 7).toFixed(2) + "deg");
      card.style.setProperty("--rx", ((.5 - py) * 7).toFixed(2) + "deg");
      card.classList.add("tilt");
    });
    card.addEventListener("pointerleave", function () {
      card.classList.remove("tilt"); card.style.removeProperty("--rx"); card.style.removeProperty("--ry");
    });
  });

  /* ---- progress: labs marked done (this browser only) */
  var doneBoxes = Array.prototype.slice.call(document.querySelectorAll("input[data-done]"));
  var prog = document.querySelector(".progress");
  function refreshProgress() {
    var total = parseInt((prog && prog.getAttribute("data-total")) || "15", 10), n = 0;
    for (var i = 0; i < total + 20; i++) { /* count keys, not boxes: lab pages have one box */ }
    var ids = (prog && prog.getAttribute("data-ids") || "").split(",").filter(Boolean);
    ids.forEach(function (id) { if (store("done:" + id) === "1") n++; });
    if (!prog) return;
    var p = ids.length ? Math.round(100 * n / ids.length) : 0;
    prog.style.setProperty("--p", p);
    var ring = prog.querySelector(".ring b"); if (ring) ring.textContent = p + "%";
    var txt = prog.querySelector(".txt"); if (txt) txt.innerHTML = "<b>" + n + " of " + ids.length + "</b> labs done" + (n === ids.length && n ? " · course complete 🎉" : n ? " · keep going" : " · tick a lab’s box when you’ve run it");
  }
  doneBoxes.forEach(function (cb) {
    var id = cb.getAttribute("data-done"), card = cb.closest(".lab-card");
    var on = store("done:" + id) === "1";
    cb.checked = on; if (card) card.classList.toggle("is-done", on);
    cb.addEventListener("change", function () {
      store("done:" + id, cb.checked ? "1" : "0");
      if (card) card.classList.toggle("is-done", cb.checked);
      refreshProgress();
    });
  });
  refreshProgress();

  /* ---- ticker: duplicate the track once so the loop is seamless */
  var track = document.querySelector(".ticker-track");
  if (track && !reduce) { track.innerHTML += track.innerHTML; }

  /* ---- "/" focuses the glossary search */
  document.addEventListener("keydown", function (e) {
    if (e.key === "/" && !/input|textarea/i.test(document.activeElement.tagName)) {
      var qq = document.getElementById("q"); if (qq) { e.preventDefault(); qq.focus(); }
    }
  });

  /* ---- hero family chips jump to the catalog and apply the matching filter */
  document.querySelectorAll("a[data-go]").forEach(function (a) {
    a.addEventListener("click", function () {
      var f = a.getAttribute("data-go"), btn = document.querySelector('.chip[data-filter="' + f + '"]');
      if (btn) btn.click();
    });
  });
})();
