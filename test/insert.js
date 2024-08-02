import t from 'tap';
import generateQuadTreeFunction from '../lib/codeGenerators/generateQuadTree.js';
import generateCreateBodyFunction from '../lib/codeGenerators/generateCreateBody.js';
import ngraphRandom from "ngraph.random";
import Graph from 'graphology';

const dimensions = 2;
const createQuadTree = generateQuadTreeFunction(dimensions);
const Body = generateCreateBodyFunction(dimensions);
const random = ngraphRandom.random(42);

t.test('insert and update update forces', function (t) {
  const g = new Graph();
  const tree = createQuadTree({}, random);
  const body = new Body();
  g.addNode("test", {body: body});
  const clone = JSON.parse(JSON.stringify(body));

  tree.insertBodies(g, {body: "body"});
  tree.updateBodyForce(body);
  t.same(body, clone, 'The body should not be changed - there are no forces acting on it');
  t.end();
});

t.test('it can get root', function (t) {
  const g = new Graph();
  const tree = createQuadTree({}, random);
  const body = new Body();
  g.addNode("test", {body: body});

  tree.insertBodies(g, {body: "body"});
  const root = tree.getRoot();
  t.ok(root, 'Root is present');
  t.equal(root.body, body, 'Body is initialized');
  t.end();
});

t.test('Two bodies repel each other', function (t) {
  const g = new Graph();
  const tree = createQuadTree({}, random);
  const bodyA = new Body(); bodyA.pos.x = 1; bodyA.pos.y = 0;
  g.addNode("test1", {body: bodyA});
  const bodyB = new Body(); bodyB.pos.x = 2; bodyB.pos.y = 0;
  g.addNode("test2", {body: bodyB});

  tree.insertBodies(g, {body: "body"});
  tree.updateBodyForce(bodyA);
  tree.updateBodyForce(bodyB);
  // based on our physical model construction forces should be equivalent, with
  // opposite sign:
  t.ok(bodyA.force.x + bodyB.force.x === 0, 'Forces should be same, with opposite sign');
  t.ok(bodyA.force.x !== 0, 'X-force for body A should not be zero');
  t.ok(bodyB.force.x !== 0, 'X-force for body B should not be zero');
  // On the other hand, our bodies should not move by Y axis:
  t.ok(bodyA.force.y === 0, 'Y-force for body A should be zero');
  t.ok(bodyB.force.y === 0, 'Y-force for body B should be zero');

  t.end();
});

t.test('Can handle two bodies at the same location', function (t) {
  const g = new Graph();
  const tree = createQuadTree({}, random);
  const bodyA = new Body();
  g.addNode("test1", {body: bodyA});
  const bodyB = new Body();
  g.addNode("test2", {body: bodyB});

  tree.insertBodies(g, {body: "body"});
  tree.updateBodyForce(bodyA);
  tree.updateBodyForce(bodyB);

  t.end();
});

t.test('it does not stuck', function(t) {
  const g = new Graph();
  let count = 60000;
  const bodies = [];

  for (let i = 0; i < count; ++i) {
    g.addNode(i, {body: new Body(Math.random(), Math.random())});
  }

  const quadTree = createQuadTree({}, random);
  quadTree.insertBodies(g, {body: "body"});

  g.forEachNode((nodeId, attr) => {
    quadTree.updateBodyForce(attr.body);
  });
  t.ok(1);
  t.end();
});
