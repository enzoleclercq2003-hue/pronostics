/* Ouverture de la pub : un « super tir » façon anime, aux couleurs du logo.
   Le ballon fonce vers nous entouré de trois arcs d'énergie qui tournoient ; à l'impact, image
   d'impact, arrêt sur image, filet qui se déforme et s'illumine ; puis les arcs se rangent face à la
   caméra et forment exactement l'anneau du logo, pendant que le ballon se dissout en lumière.
   Style original (aucun personnage ni technique repris d'une œuvre existante).
   Dépend de vendor/three.min.js (Three.js r149, licence MIT). Expose window.Kick. */
(function () {
  const T3 = THREE, R = 0.11, GOAL_W = 7.32, GOAL_H = 2.44, DEPTH = 1.7, HALF = GOAL_W / 2, W = 390, H = 844;
  const HIT = new T3.Vector3(2.35, 1.62, DEPTH), START = new T3.Vector3(-2.2, 0.55, -23), SPEED = 31;
  const CAM0 = new T3.Vector3(1.85, 0.62, 5.5), LOOK0 = new T3.Vector3(2.5, 1.62, 0.5);
  // couleurs du logo, converties en espace linéaire pour ressortir à l'écran exactement comme dans le logo
  const BLUE = new T3.Color(0x2997ff).convertSRGBToLinear(), VIOLET = new T3.Color(0x8f7bff).convertSRGBToLinear(), PINK = new T3.Color(0xd86bff).convertSRGBToLinear(), WHITE = new T3.Color(0xffffff);
  const RING_R = 0.215, TUBE = RING_R * 5.75 / 30;                 // mêmes proportions que l'anneau du logo
  const ARCS = [[-90, 170], [112, 56], [200, 38]];                 // (départ, étendue) en degrés, comme dans le logo
  let renderer, scene, camera, ball, state, netGeo, netPos, netCol, raf = 0, cb = {}, fx, fxc, aura, aura2, ringGroup, pivots = [], sparks, sparkData, waves = [];

  const smooth = x => { x = Math.min(1, Math.max(0, x)); return x * x * (3 - 2 * x); };
  const lerp = (a, b, k) => a + (b - a) * k;
  const grad = (t, out) => t < .55 ? out.copy(BLUE).lerp(VIOLET, t / .55) : out.copy(VIOLET).lerp(PINK, (t - .55) / .45);

  function toonMap() { const d = new Uint8Array([70, 150, 235, 255]), t = new T3.DataTexture(d, 4, 1, T3.RedFormat); t.minFilter = t.magFilter = T3.NearestFilter; t.needsUpdate = true; return t; }
  function glowTexture() {
    const c = document.createElement('canvas'); c.width = c.height = 128; const x = c.getContext('2d'), g = x.createRadialGradient(64, 64, 0, 64, 64, 64);
    g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(.25, 'rgba(255,255,255,.6)'); g.addColorStop(.6, 'rgba(255,255,255,.14)'); g.addColorStop(1, 'rgba(255,255,255,0)'); x.fillStyle = g; x.fillRect(0, 0, 128, 128);
    return new T3.CanvasTexture(c);
  }

  // ---- ballon : icosaèdre tronqué, ombrage « dessin animé » et contour noir ----
  function buildBall(tm) {
    const phi = (1 + Math.sqrt(5)) / 2, V = [];
    for (const a of [-1, 1]) for (const b of [-1, 1]) { V.push([0, a, b * phi]); V.push([a, b * phi, 0]); V.push([b * phi, 0, a]); }
    const P = V.map(p => new T3.Vector3(p[0], p[1], p[2])), near = (i, j) => Math.abs(P[i].distanceToSquared(P[j]) - 4) < 1e-6;
    const nb = P.map((_, i) => P.map((_, j) => j).filter(j => j !== i && near(i, j))), polys = [];
    for (let i = 0; i < 12; i++) { const c = P[i], u = new T3.Vector3().crossVectors(c, Math.abs(c.y) < 1 ? new T3.Vector3(0, 1, 0) : new T3.Vector3(1, 0, 0)).normalize(), w = new T3.Vector3().crossVectors(c, u).normalize();
      polys.push({pent: true, pts: nb[i].map(j => c.clone().lerp(P[j], 1 / 3)).sort((a, b) => Math.atan2(a.dot(w), a.dot(u)) - Math.atan2(b.dot(w), b.dot(u)))}); }
    for (let i = 0; i < 12; i++) for (const j of nb[i]) if (j > i) for (const k of nb[j]) if (k > j && near(i, k)) { const [a, b, c] = [P[i], P[j], P[k]];
      polys.push({pent: false, pts: [a.clone().lerp(b, 1 / 3), a.clone().lerp(b, 2 / 3), b.clone().lerp(c, 1 / 3), b.clone().lerp(c, 2 / 3), c.clone().lerp(a, 1 / 3), c.clone().lerp(a, 2 / 3)]}); }
    const out = {true: [], false: []};
    const tri = (a, b, c, d, arr) => { if (d === 0) { for (const p of [a, b, c]) { const n = p.clone().normalize(); arr.push(n.x * R, n.y * R, n.z * R); } return; }
      const ab = a.clone().add(b).multiplyScalar(.5), bc = b.clone().add(c).multiplyScalar(.5), ca = c.clone().add(a).multiplyScalar(.5);
      tri(a, ab, ca, d - 1, arr); tri(ab, b, bc, d - 1, arr); tri(ca, bc, c, d - 1, arr); tri(ab, bc, ca, d - 1, arr); };
    for (const poly of polys) { const c = poly.pts.reduce((s, p) => s.add(p), new T3.Vector3()).multiplyScalar(1 / poly.pts.length); let pts = poly.pts.map(p => c.clone().lerp(p, 0.9));
      if (new T3.Vector3().crossVectors(pts[1].clone().sub(pts[0]), pts[2].clone().sub(pts[0])).dot(c) < 0) pts.reverse();
      for (let i = 0; i < pts.length; i++) tri(c, pts[i], pts[(i + 1) % pts.length], 3, out[poly.pent]); }
    const mesh = (arr, mat) => { const g = new T3.BufferGeometry(); g.setAttribute('position', new T3.Float32BufferAttribute(arr, 3));
      const n = new Float32Array(arr.length); for (let i = 0; i < arr.length; i += 3) { const l = Math.hypot(arr[i], arr[i + 1], arr[i + 2]); n[i] = arr[i] / l; n[i + 1] = arr[i + 1] / l; n[i + 2] = arr[i + 2] / l; }
      g.setAttribute('normal', new T3.BufferAttribute(n, 3)); return new T3.Mesh(g, mat); };
    const grp = new T3.Group();
    grp.add(new T3.Mesh(new T3.SphereGeometry(R * 0.985, 40, 28), new T3.MeshBasicMaterial({color: 0x05050c})));
    grp.add(mesh(out[false], new T3.MeshToonMaterial({color: 0xffffff, gradientMap: tm})));
    grp.add(mesh(out[true], new T3.MeshToonMaterial({color: 0x2b3a8f, gradientMap: tm})));
    grp.add(new T3.Mesh(new T3.SphereGeometry(R * 1.075, 40, 28), new T3.MeshBasicMaterial({color: 0x04040a, side: T3.BackSide})));   // contour
    return grp;
  }

  // ---- arcs d'énergie = les trois segments du logo ----
  function buildRings() {
    ringGroup = new T3.Group(); pivots = [];
    ARCS.forEach(([a, e], i) => {
      const arc = e * Math.PI / 180, geo = new T3.TorusGeometry(RING_R, TUBE, 14, Math.max(10, Math.round(e / 3)), arc), pos = geo.attributes.position, col = new Float32Array(pos.count * 3), c = new T3.Color();
      for (let k = 0; k < pos.count; k++) { const th = Math.atan2(pos.getY(k), pos.getX(k)), t = ((th < -0.01 ? th + 2 * Math.PI : th) / arc); (i === 0 ? grad(Math.min(1, Math.max(0, t)), c) : c.copy(WHITE)); col.set([c.r, c.g, c.b], k * 3); }
      geo.setAttribute('color', new T3.BufferAttribute(col, 3));
      // « transparent » même pour l'arc opaque : il est ainsi dessiné après les halos, donc par-dessus
      const mat = new T3.MeshBasicMaterial({vertexColors: true, transparent: true, opacity: [1, .55, .26][i], depthTest: false, depthWrite: false});
      const m = new T3.Mesh(geo, mat), holder = new T3.Group(); m.renderOrder = 10;
      for (const th of [0, arc]) { const s = new T3.Mesh(new T3.SphereGeometry(TUBE, 14, 10), new T3.MeshBasicMaterial({color: i === 0 ? grad(th === 0 ? 0 : 1, new T3.Color()) : 0xffffff, transparent: true, opacity: [1, .55, .26][i], depthTest: false, depthWrite: false}));
        s.position.set(Math.cos(th) * RING_R, Math.sin(th) * RING_R, 0); s.renderOrder = 10; holder.add(s); }
      holder.add(m); holder.rotation.z = -(a + e) * Math.PI / 180;            // même orientation que dans le logo (angles horaires en SVG)
      const pivot = new T3.Group(); pivot.add(holder); pivot.userData = {a: 7 + i * 3.1, b: 11 - i * 2.3, c: 5 + i * 4.7}; pivots.push(pivot); ringGroup.add(pivot);
    });
    return ringGroup;
  }

  function tube(a, b, r, mat) { const m = new T3.Mesh(new T3.CylinderGeometry(r, r, a.distanceTo(b), 20), mat); m.position.copy(a).lerp(b, .5); m.quaternion.setFromUnitVectors(new T3.Vector3(0, 1, 0), b.clone().sub(a).normalize()); return m; }
  function gridLines(o, du, dv, nu, nv) { const pts = [];
    for (let i = 0; i <= nu; i++) { const a = o.clone().addScaledVector(du, i / nu); pts.push(a, a.clone().add(dv)); }
    for (let j = 0; j <= nv; j++) { const a = o.clone().addScaledVector(dv, j / nv); pts.push(a, a.clone().add(du)); }
    return new T3.BufferGeometry().setFromPoints(pts); }

  const COLS = 54, ROWS = 17, NET_H = 2.36, NSPARK = 150;
  function buildScene() {
    scene = new T3.Scene(); scene.background = new T3.Color(0x05040d); scene.fog = new T3.FogExp2(0x07061a, 0.03);
    camera = new T3.PerspectiveCamera(46, W / H, 0.05, 400);
    const tm = toonMap(), glow = glowTexture();
    scene.add(new T3.AmbientLight(0xbfc8ff, .55)); const sun = new T3.DirectionalLight(0xffffff, 1.25); sun.position.set(-3, 6, 9); scene.add(sun);
    // ciel de nuit en dégradé (violet à l'horizon)
    const sky = new T3.SphereGeometry(300, 32, 96), sc = new Float32Array(sky.attributes.position.count * 3), c1 = new T3.Color(0x03030a), c2 = new T3.Color(0x160f3c), tmp = new T3.Color();
    for (let i = 0; i < sky.attributes.position.count; i++) { const y = sky.attributes.position.getY(i) / 300; tmp.copy(c2).lerp(c1, smooth(Math.abs(y) * 4.5)); sc.set([tmp.r, tmp.g, tmp.b], i * 3); }
    sky.setAttribute('color', new T3.BufferAttribute(sc, 3)); scene.add(new T3.Mesh(sky, new T3.MeshBasicMaterial({vertexColors: true, side: T3.BackSide, fog: false, depthWrite: false})));
    // pelouse stylisée : bandes sombres bleutées, sans texture réaliste
    const gc = document.createElement('canvas'); gc.width = 8; gc.height = 256; const gx = gc.getContext('2d'); for (let i = 0; i < 8; i++) { gx.fillStyle = i % 2 ? '#0d2233' : '#0a1b29'; gx.fillRect(0, i * 32, 8, 32); }
    const gt = new T3.CanvasTexture(gc); gt.wrapS = gt.wrapT = T3.RepeatWrapping; gt.repeat.set(1, 5); gt.magFilter = T3.NearestFilter; gt.encoding = T3.sRGBEncoding;
    const ground = new T3.Mesh(new T3.PlaneGeometry(160, 160), new T3.MeshBasicMaterial({map: gt})); ground.rotation.x = -Math.PI / 2; scene.add(ground);
    const line = new T3.Mesh(new T3.PlaneGeometry(60, .14), new T3.MeshBasicMaterial({color: 0xcfd6ff})); line.rotation.x = -Math.PI / 2; line.position.y = .004; scene.add(line);
    const white = new T3.MeshToonMaterial({color: 0xffffff, gradientMap: tm}), steel = new T3.MeshToonMaterial({color: 0x8a90b8, gradientMap: tm});
    scene.add(tube(new T3.Vector3(-HALF, 0, 0), new T3.Vector3(-HALF, GOAL_H, 0), .065, white), tube(new T3.Vector3(HALF, 0, 0), new T3.Vector3(HALF, GOAL_H, 0), .065, white),
      tube(new T3.Vector3(-HALF - .065, GOAL_H, 0), new T3.Vector3(HALF + .065, GOAL_H, 0), .065, white));
    for (const sx of [-1, 1]) scene.add(tube(new T3.Vector3(sx * HALF, NET_H, DEPTH), new T3.Vector3(sx * HALF, 0, DEPTH), .022, steel), tube(new T3.Vector3(sx * HALF, GOAL_H, 0), new T3.Vector3(sx * HALF, NET_H, DEPTH), .022, steel));
    scene.add(tube(new T3.Vector3(-HALF, NET_H, DEPTH), new T3.Vector3(HALF, NET_H, DEPTH), .022, steel));
    const netMat = new T3.LineBasicMaterial({color: 0xaab4ff, transparent: true, opacity: .34});
    scene.add(new T3.LineSegments(gridLines(new T3.Vector3(-HALF, GOAL_H, 0), new T3.Vector3(GOAL_W, 0, 0), new T3.Vector3(0, NET_H - GOAL_H, DEPTH), COLS, 12), netMat));
    for (const sx of [-1, 1]) scene.add(new T3.LineSegments(gridLines(new T3.Vector3(sx * HALF, 0, 0), new T3.Vector3(0, 0, DEPTH), new T3.Vector3(0, NET_H, 0), 12, ROWS), netMat));
    const nseg = COLS * (ROWS + 1) + (COLS + 1) * ROWS; netGeo = new T3.BufferGeometry(); netPos = new Float32Array(nseg * 6); netCol = new Float32Array(nseg * 6);
    netGeo.setAttribute('position', new T3.BufferAttribute(netPos, 3)); netGeo.setAttribute('color', new T3.BufferAttribute(netCol, 3));
    scene.add(new T3.LineSegments(netGeo, new T3.LineBasicMaterial({vertexColors: true, transparent: true, opacity: .9})));
    // projecteurs lointains, aux teintes du logo
    const n = 90, pos = new Float32Array(n * 3), col = new Float32Array(n * 3); let seed = 11; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let i = 0; i < n; i++) { const tier = i % 3; pos.set([(rnd() - .5) * 190, 12 + tier * 8 + rnd() * 3, -95 - tier * 6], i * 3); grad(rnd(), tmp).lerp(WHITE, .35); col.set([tmp.r, tmp.g, tmp.b], i * 3); }
    const bg = new T3.BufferGeometry(); bg.setAttribute('position', new T3.BufferAttribute(pos, 3)); bg.setAttribute('color', new T3.BufferAttribute(col, 3));
    scene.add(new T3.Points(bg, new T3.PointsMaterial({size: 13, map: glow, vertexColors: true, transparent: true, opacity: .32, depthWrite: false, blending: T3.AdditiveBlending, fog: false})));
    // ballon, halo, arcs, étincelles, ondes de choc
    ball = buildBall(tm); scene.add(ball);
    const sprite = (color, s) => { const sp = new T3.Sprite(new T3.SpriteMaterial({map: glow, color, transparent: true, depthWrite: false, depthTest: true, blending: T3.AdditiveBlending, fog: false})); sp.scale.setScalar(s); sp.renderOrder = 5; scene.add(sp); return sp; };
    aura = sprite(0x8f7bff, 1.0); aura2 = sprite(0x2997ff, 1.9);
    scene.add(buildRings());
    const sg = new T3.BufferGeometry(); sg.setAttribute('position', new T3.BufferAttribute(new Float32Array(NSPARK * 3), 3)); sg.setAttribute('color', new T3.BufferAttribute(new Float32Array(NSPARK * 3), 3));
    sparks = new T3.Points(sg, new T3.PointsMaterial({size: .16, map: glow, vertexColors: true, transparent: true, depthWrite: false, depthTest: false, blending: T3.AdditiveBlending, fog: false})); sparks.renderOrder = 6; sparks.frustumCulled = false; scene.add(sparks);
    for (let i = 0; i < 3; i++) { const wv = new T3.Mesh(new T3.RingGeometry(.93, 1, 72), new T3.MeshBasicMaterial({color: [0xffffff, 0x8f7bff, 0x2997ff][i], transparent: true, opacity: 0, depthWrite: false, depthTest: false, blending: T3.AdditiveBlending, side: T3.DoubleSide, fog: false})); wv.renderOrder = 4; scene.add(wv); waves.push(wv); }
  }

  // ---- état et physique ----
  function reset() {
    const p = [], prev = [];
    for (let j = 0; j <= ROWS; j++) for (let i = 0; i <= COLS; i++) { const v = new T3.Vector3(-HALF + GOAL_W * i / COLS, NET_H * j / ROWS, DEPTH); p.push(v); prev.push(v.clone()); }
    sparkData = Array.from({length: NSPARK}, () => ({life: 0, max: 1, pos: new T3.Vector3(0, -50, 0), vel: new T3.Vector3(), t: 0}));
    state = {T: 0, sim: 0, impact: null, logo: false, said: {}, p, prev, pos: START.clone(), vel: HIT.clone().sub(START).normalize().multiplyScalar(SPEED), axis: new T3.Vector3(.35, .9, .2).normalize(), next: 0, frame: 0};
    ball.position.copy(state.pos); ball.quaternion.identity(); ball.scale.setScalar(1); waves.forEach(w => w.material.opacity = 0);
  }
  const idx = (i, j) => j * (COLS + 1) + i, pinned = (i, j) => i === 0 || i === COLS || j === 0 || j === ROWS, RX = GOAL_W / COLS, RY = NET_H / ROWS;
  function stepNet(h) {
    const {p, prev, pos} = state;
    for (let j = 1; j < ROWS; j++) for (let i = 1; i < COLS; i++) { const k = idx(i, j), a = p[k], b = prev[k], vx = (a.x - b.x) * .992, vy = (a.y - b.y) * .992, vz = (a.z - b.z) * .992; b.copy(a); a.x += vx; a.y += vy - 3.2 * h * h; a.z += vz; }
    const link = (k1, k2, rest, f1, f2) => { const a = p[k1], b = p[k2], dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z, d = Math.hypot(dx, dy, dz) || 1e-6; if (d < rest) return;
      const diff = (d - rest) / d, s1 = f1 ? 0 : f2 ? 1 : .5, s2 = f2 ? 0 : f1 ? 1 : .5; a.x += dx * diff * s1; a.y += dy * diff * s1; a.z += dz * diff * s1; b.x -= dx * diff * s2; b.y -= dy * diff * s2; b.z -= dz * diff * s2; };
    const rr = R + .02;
    for (let it = 0; it < 12; it++) {
      for (let j = 0; j <= ROWS; j++) for (let i = 0; i < COLS; i++) link(idx(i, j), idx(i + 1, j), RX * 1.03, pinned(i, j), pinned(i + 1, j));
      for (let j = 0; j < ROWS; j++) for (let i = 0; i <= COLS; i++) link(idx(i, j), idx(i, j + 1), RY * 1.03, pinned(i, j), pinned(i, j + 1));
      // le ballon vient du terrain : le filet reste plaqué sur sa face avant (sinon les fils glissent autour et il traverse)
      for (let j = 1; j < ROWS; j++) for (let i = 1; i < COLS; i++) { const a = p[idx(i, j)], dx = a.x - pos.x, dy = a.y - pos.y, r2 = dx * dx + dy * dy; if (r2 < rr * rr) { const f = pos.z + Math.sqrt(rr * rr - r2); if (a.z < f) a.z = f; } }
    }
  }
  function stepBall(h) {
    const s = state;
    if (s.impact == null) { s.pos.addScaledVector(s.vel, h); if (s.pos.z >= DEPTH - R) { s.impact = s.T; burst(); if (cb.impact) cb.impact(); } return; }
    const w = 22, z0 = DEPTH + .16; s.vel.z += (-w * w * (s.pos.z - z0) - 2 * .72 * w * s.vel.z) * h;
    const k = Math.exp(-10 * h); s.vel.x *= k; s.vel.y = s.vel.y * k - 1.6 * h; s.pos.addScaledVector(s.vel, h);
  }
  function emit(pos, vel, life, t) { const d = sparkData[state.next = (state.next + 1) % NSPARK]; d.pos.copy(pos); d.vel.copy(vel); d.life = d.max = life; d.t = t; }
  function burst() { const s = state; let seed = 5; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let i = 0; i < 80; i++) { const a = rnd() * Math.PI * 2, sp = 1.5 + rnd() * 5.5; emit(s.pos, new T3.Vector3(Math.cos(a) * sp, Math.sin(a) * sp, 1.5 + rnd() * 3), .5 + rnd() * .6, rnd()); } }

  const qWild = new T3.Quaternion(), qId = new T3.Quaternion(), eul = new T3.Euler(), tmpC = new T3.Color(), tmpV = new T3.Vector3();
  function advance(dt) {
    const s = state; s.T += dt; s.frame++;
    const since = s.impact == null ? -1 : s.T - s.impact;
    // arrêt sur image à l'impact, puis ralenti
    const scale = since < 0 ? 1 : since < .12 ? 0 : lerp(.55, .14, smooth((since - .12) / .25));
    const sub = 10, h = dt * scale / sub; s.sim += dt * scale;
    for (let i = 0; i < sub && h > 0; i++) { stepBall(h); stepNet(h); }
    const since2 = s.impact == null ? -1 : s.T - s.impact;
    ball.position.copy(s.pos); ball.rotateOnWorldAxis(s.axis, (since2 < 0 ? 34 : since2 < .12 ? 0 : lerp(26, 5, smooth((since2 - .12) / .7))) * dt);
    const gone = smooth((since2 - .95) / .5); ball.scale.setScalar(Math.max(.001, 1 - gone));                      // le ballon se dissout en lumière
    // caméra : contre-plongée derrière le filet, secousse à l'impact, puis face au ballon
    const k = since2 < 0 ? 0 : smooth((since2 - .12) / 1.15), amp = since2 < 0 ? .012 : .12 * Math.exp(-Math.max(0, since2 - .12) * 7);
    const want = s.pos.clone().add(new T3.Vector3(0, 0, 3.4)), cam = CAM0.clone().lerp(want, k), look = LOOK0.clone().lerp(s.pos, smooth((since2 + .2) / .6));
    cam.x += Math.sin(s.T * 97) * amp; cam.y += Math.cos(s.T * 83) * amp;
    camera.position.copy(cam); camera.lookAt(look); camera.fov = lerp(46, 40, k); camera.updateProjectionMatrix(); camera.updateMatrixWorld();
    // halos
    // halos : placés derrière le ballon (il reste lisible), éclat bref à l'impact et quand le ballon se dissout, puis extinction
    const pulse = 1 + .1 * Math.sin(s.T * 40), flash = since2 < 0 ? 0 : Math.exp(-Math.max(0, since2 - .12) * 6) * 1.6 + Math.sin(Math.PI * gone) * .7, fade = 1 - smooth((since2 - 1.25) / .3);
    const back = tmpV.copy(s.pos).sub(camera.position).normalize().multiplyScalar(R * 1.6).add(s.pos);
    aura.position.copy(back); aura2.position.copy(back); aura.scale.setScalar((.62 + flash) * pulse); aura2.scale.setScalar(1.25 + flash * 1.4);
    aura.material.opacity = (.6 + Math.min(.3, flash * .3)) * fade; aura2.material.opacity = (.26 + Math.min(.4, flash * .25)) * fade;
    // arcs : tournoiement libre en vol, puis alignement face caméra = l'anneau du logo
    const align = since2 < 0 ? 0 : smooth((since2 - .3) / .85);
    ringGroup.position.copy(s.pos); ringGroup.quaternion.copy(camera.quaternion); ringGroup.scale.setScalar(since2 < 0 ? 1 : 1 + .35 * Math.exp(-since2 * 6));
    pivots.forEach(p => { const u = p.userData; eul.set(s.T * u.a, s.T * u.b, s.T * u.c); qWild.setFromEuler(eul); p.quaternion.copy(qWild).slerp(qId, align); });
    // traînée d'étincelles en vol
    if (since2 < 0) for (let i = 0; i < 5; i++) { const j = () => (Math.random() - .5) * .16; emit(tmpV.copy(s.pos).add(new T3.Vector3(j(), j(), j())), new T3.Vector3(j() * 6, j() * 6, -3), .32 + Math.random() * .2, Math.random()); }
    const sp = sparks.geometry.attributes.position, scol = sparks.geometry.attributes.color;
    sparkData.forEach((d, i) => { if (d.life > 0) { d.life -= dt * (since2 >= 0 && since2 < .12 ? 0 : 1); d.pos.addScaledVector(d.vel, dt * (scale > 0 ? Math.max(scale, .3) : 0)); d.vel.multiplyScalar(1 - 2.5 * dt); }
      const a = Math.max(0, d.life / d.max); grad(d.t, tmpC).lerp(WHITE, a * .6).multiplyScalar(a); sp.setXYZ(i, d.pos.x, d.pos.y, d.life > 0 ? d.pos.z : -500); scol.setXYZ(i, tmpC.r, tmpC.g, tmpC.b); });
    sp.needsUpdate = scol.needsUpdate = true;
    // ondes de choc
    waves.forEach((w, i) => { const t = since2 - .12 - i * .09; if (since2 < 0 || t < 0) { w.material.opacity = 0; return; } w.position.copy(s.pos); w.quaternion.copy(camera.quaternion); w.scale.setScalar(.2 + t * 3.4); w.material.opacity = Math.max(0, .9 - t * 1.5); });
    if (!s.logo && since2 >= 1.6) { s.logo = true; if (cb.logo) cb.logo(screenRing()); }
  }
  function project(v) { const c = v.clone().project(camera); return {x: (c.x + 1) / 2 * W, y: (1 - c.y) / 2 * H}; }
  function screenRing() { const c = project(state.pos), up = new T3.Vector3(0, 1, 0).applyQuaternion(camera.quaternion), e = project(state.pos.clone().addScaledVector(up, RING_R)); return {x: c.x, y: c.y, r: Math.abs(e.y - c.y)}; }

  function drawFx() {                        // lignes de vitesse et image d'impact, dessinées par-dessus la 3D
    if (!fxc) return; const s = state, c = project(s.pos), since = s.impact == null ? -1 : s.T - s.impact; fxc.clearRect(0, 0, W, H);
    let seed = s.frame * 7919 + 13; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const rays = (n, r0, r1, wmax, style) => { fxc.fillStyle = style; for (let i = 0; i < n; i++) { const a = rnd() * Math.PI * 2, w = (.004 + rnd() * wmax), ra = r0 * (.7 + rnd() * .9), rb = r1;
      fxc.beginPath(); fxc.moveTo(c.x + Math.cos(a) * ra, c.y + Math.sin(a) * ra); fxc.lineTo(c.x + Math.cos(a - w) * rb, c.y + Math.sin(a - w) * rb); fxc.lineTo(c.x + Math.cos(a + w) * rb, c.y + Math.sin(a + w) * rb); fxc.fill(); } };
    if (since < 0) { const k = smooth(s.T / .35) * .8; rays(46, 150, 1100, .02, `rgba(255,255,255,${(.55 * k).toFixed(3)})`); rays(16, 190, 1100, .03, `rgba(143,123,255,${(.5 * k).toFixed(3)})`); }
    else if (since < .12) {                 // image d'impact : fond clair, rayons sombres, puis l'inverse
      const dark = since > .06; fxc.fillStyle = dark ? 'rgba(6,5,20,.92)' : 'rgba(255,255,255,.94)'; fxc.fillRect(0, 0, W, H);
      rays(60, 34, 1100, .035, dark ? 'rgba(255,255,255,.95)' : 'rgba(10,8,34,.95)');
      const g = fxc.createRadialGradient(c.x, c.y, 0, c.x, c.y, 130); g.addColorStop(0, 'rgba(255,255,255,1)'); g.addColorStop(.5, 'rgba(160,140,255,.85)'); g.addColorStop(1, 'rgba(41,151,255,0)'); fxc.fillStyle = g; fxc.fillRect(0, 0, W, H);
    } else if (since < .7) { const k = 1 - (since - .12) / .58; rays(34, 120 + (1 - k) * 260, 1100, .02, `rgba(200,180,255,${(.5 * k * k).toFixed(3)})`); }
  }
  function draw() {
    const {p, pos, impact, T} = state; let o = 0, q = 0; const since = impact == null ? -1 : T - impact, heat = since < 0 ? 0 : Math.exp(-Math.max(0, since - .12) * 1.8);
    const seg = (a, b) => { for (const v of [a, b]) { netPos[o++] = v.x; netPos[o++] = v.y; netPos[o++] = v.z;
      const d = Math.hypot(v.x - pos.x, v.y - pos.y), e = heat * Math.exp(-d * d / .9); grad(Math.min(1, d / 1.4), tmpC).lerp(WHITE, e * .5); const base = .34 + e * .9;
      netCol[q++] = lerp(.62, tmpC.r, Math.min(1, e * 1.4)) * base; netCol[q++] = lerp(.66, tmpC.g, Math.min(1, e * 1.4)) * base; netCol[q++] = lerp(1, tmpC.b, Math.min(1, e * 1.4)) * base; } };
    for (let j = 0; j <= ROWS; j++) for (let i = 0; i < COLS; i++) seg(p[idx(i, j)], p[idx(i + 1, j)]);
    for (let j = 0; j < ROWS; j++) for (let i = 0; i <= COLS; i++) seg(p[idx(i, j)], p[idx(i, j + 1)]);
    netGeo.attributes.position.needsUpdate = netGeo.attributes.color.needsUpdate = true; renderer.render(scene, camera); drawFx();
  }

  window.Kick = {
    init(canvas, fxCanvas, keepBuffer) {    // keepBuffer : garde l'image pour les captures d'écran (mode image fixe)
      if (renderer) return true;
      try { renderer = new T3.WebGLRenderer({canvas, antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: !!keepBuffer}); } catch (e) { return false; }
      renderer.outputEncoding = T3.sRGBEncoding; fx = fxCanvas; if (fx) { fx.width = W * 2; fx.height = H * 2; fxc = fx.getContext('2d'); fxc.scale(2, 2); }
      this.resize(1); buildScene(); reset(); return true;
    },
    resize(k) { if (renderer) { renderer.setPixelRatio(Math.min(3, (window.devicePixelRatio || 1) * Math.max(1, k))); renderer.setSize(W, H, false); } },
    start(callbacks) { this.stop(); cb = callbacks || {}; reset(); let last = performance.now();
      const loop = now => { const dt = Math.min(.05, (now - last) / 1000); last = now; advance(dt); draw(); raf = requestAnimationFrame(loop); }; raf = requestAnimationFrame(loop); },
    stop() { cancelAnimationFrame(raf); raf = 0; },
    clearFx() { if (fxc) fxc.clearRect(0, 0, W, H); },
    renderAt(ms, callbacks) { this.stop(); cb = callbacks || {}; reset(); for (let t = 0; t < ms / 1000; t += 1 / 60) advance(1 / 60); draw(); }   // image fixe à un instant donné
  };
})();
