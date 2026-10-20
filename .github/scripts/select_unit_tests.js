// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

const {execFileSync} = require('node:child_process');
const {createHash} = require('node:crypto');
const fs = require('node:fs');

// Keep cache restores sequential: their path is part of the existing cache version.
module.exports = async ({core, cache, artifact}) => {
  const {BUCKETS, RECIPE_PLATFORM, SOURCE_SHA, TARGET_BRANCH, PR_FILES_DIR,
    GITHUB_RUN_ID, GITHUB_RUN_ATTEMPT, RUNNER_TEMP} = process.env;
  if (TARGET_BRANCH !== 'main') throw new Error('Testmon selection requires main');

  const matrix = [];
  const summary = [['Bucket', 'Mode', 'GPU required']];
  const python = (script, args) => {
    const output = execFileSync('python', [`tests/unit_tests/${script}`, ...args], {
      encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'],
    });
    core.info(output.trim());
    return Object.fromEntries(output.trim().split('\n').map(line => line.split('=')));
  };

  for (const entry of JSON.parse(BUCKETS)) {
    const {bucket} = entry;
    core.startGroup(bucket);
    const identity = python('testmon_cache.py', [
      'identity', '--bucket', bucket, '--platform', RECIPE_PLATFORM,
      '--output', `${RUNNER_TEMP}/unit-testmon-identity.json`,
    ]);
    fs.rmSync('assets_dir/testmon', {recursive: true, force: true});
    let matchedKey = '';
    try {
      matchedKey = await cache.restoreCache(
        ['assets_dir/testmon'],
        `${identity.cache_prefix}lookup-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}`,
        [identity.cache_prefix],
      ) || '';
    } catch (error) {
      core.warning(`Testmon cache restore failed for ${bucket}: ${error.message}`);
    }
    const plan = 'assets_dir/testmon-plan/plan.json';
    const result = python('testmon_selector.py', [
      'select', '--cache-dir', 'assets_dir/testmon', '--bucket', bucket,
      '--platform', RECIPE_PLATFORM, '--source-sha', SOURCE_SHA,
      '--matched-key', matchedKey, '--pr-files-dir', PR_FILES_DIR, '--output', plan,
    ]);
    if (!['selected', 'full'].includes(result.mode) ||
        !['true', 'false'].includes(result.has_tests) ||
        (result.mode === 'full' && result.has_tests !== 'true')) {
      throw new Error(`Invalid Testmon selection result for ${bucket}`);
    }
    let planArtifact = '';
    if (result.mode === 'selected') {
      const hash = createHash('sha256').update(bucket).digest('hex').slice(0, 16);
      planArtifact = `unit-test-plan-${RECIPE_PLATFORM}-${hash}-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}`;
      // Upload empty plans too: the coverage gate uses them as skip evidence.
      await artifact.uploadArtifact(planArtifact, [plan], 'assets_dir/testmon-plan');
    }
    if (result.has_tests === 'true') {
      matrix.push({...entry, mode: result.mode === 'selected' ? 'preselected' : 'full',
        plan_artifact: planArtifact});
    }
    summary.push([bucket, result.mode, result.has_tests]);
    core.endGroup();
  }
  core.setOutput('matrix', JSON.stringify(matrix));
  core.setOutput('has_tests', String(matrix.length > 0));
  await core.summary.addHeading('Unit Testmon CPU selection').addTable(summary).write();
};
