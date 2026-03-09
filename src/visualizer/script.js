let simData = null;
let degreeChart = null;
let powerlawChart = null;
let surveyChart = null;
let opinionChangesChart = null;
let semanticDriftChart = null;
let simulationGraph = null;
let currentStep = 0;
let maxStep = 0;
let surveySteps = [];
let selectedNode = null;
let selectedMessageId = null;
let controlsInitialized = false;

document.getElementById('fileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (!file) {
        return;
    }

    const reader = new FileReader();
    reader.onload = function(loadEvent) {
        try {
            simData = JSON.parse(loadEvent.target.result);
            selectedNode = null;
            selectedMessageId = null;
            processData();
            initVisualizations();
        } catch (error) {
            console.error('Error parsing JSON:', error);
            alert('Invalid JSON file.');
        }
    };
    reader.readAsText(file);
});

function processData() {
    const degrees = new Array(simData.nodes.length).fill(0);
    simData.edges.forEach(edge => {
        degrees[edge.source] += 1;
        degrees[edge.target] += 1;
    });
    simData.nodes.forEach((node, index) => {
        node.degree = degrees[index];
    });

    maxStep = 0;
    if (simData.survey_results.length > 0) {
        maxStep = Math.max(maxStep, ...simData.survey_results.map(result => result.step));
    }
    if (simData.observations.length > 0) {
        maxStep = Math.max(maxStep, ...simData.observations.map(observation => observation.step));
    }

    surveySteps = simData.survey_results.map(result => result.step).sort((left, right) => left - right);
    if (surveySteps.length === 0) {
        surveySteps = [0];
    }

    currentStep = surveySteps[0];
    const slider = document.getElementById('timeSlider');
    slider.max = Math.max(surveySteps.length - 1, 0);
    slider.value = 0;
    document.getElementById('timeValue').innerText = currentStep;

    if (!controlsInitialized) {
        initializeControls();
        controlsInitialized = true;
    }

    updateInfluenceSummary();
}

function initializeControls() {
    document.getElementById('timeSlider').addEventListener('input', function() {
        const index = parseInt(this.value, 10);
        currentStep = surveySteps[index] ?? 0;
        document.getElementById('timeValue').innerText = currentStep;
        updateGraph();
        drawMessageReach();
        updateNodeDetails(selectedNode);
    });

    document.getElementById('toggleLatestVoteColor').addEventListener('change', updateGraph);
    document.getElementById('toggleChangedMinds').addEventListener('change', updateGraph);
    document.getElementById('toggleOpinionChangeColor').addEventListener('change', updateGraph);
    document.getElementById('toggleNodeSize').addEventListener('change', updateGraph);
    document.getElementById('toggleNodeSizeViews').addEventListener('change', updateGraph);
    document.getElementById('toggleNewsLines').addEventListener('change', updateSurveyChart);
    document.querySelectorAll('input[name="surveyYAxis"]').forEach(radio => {
        radio.addEventListener('change', updateSurveyChart);
    });

    document.getElementById('influenceThreshold').addEventListener('input', function() {
        document.getElementById('influenceThresholdValue').innerText = getInfluenceThreshold().toFixed(2);
        updateInfluenceGraph();
        if (selectedMessageId !== null) {
            updateMessageDetails(selectedMessageId);
            updateCascadeTree(selectedMessageId);
        }
    });

    document.getElementById('influenceNodeLimit').addEventListener('change', updateInfluenceGraph);
}

function initVisualizations() {
    drawDegreeDistribution();
    drawPowerlawDistribution();
    drawSurveyChart();
    drawOpinionChangesDistribution();
    drawSemanticDriftChart();
    drawImpactTable();
    drawMessageReach();
    initGraph();
    updateGraph();
    updateInfluenceGraph();
    updateMessageDetails(selectedMessageId);
    updateCascadeTree(selectedMessageId);
}

function getInfluenceAnalysis() {
    return simData && simData.influence_analysis ? simData.influence_analysis : null;
}

function getInfluenceMessages() {
    const analysis = getInfluenceAnalysis();
    return analysis ? (analysis.messages || []) : [];
}

function getInfluenceMessageMap() {
    const messageMap = new Map();
    getInfluenceMessages().forEach(message => {
        messageMap.set(message.id, message);
    });
    return messageMap;
}

function getInfluenceThreshold() {
    const sliderValue = parseInt(document.getElementById('influenceThreshold').value, 10);
    return sliderValue / 100;
}

function updateInfluenceSummary() {
    const analysis = getInfluenceAnalysis();
    const summaryDiv = document.getElementById('influence-summary');
    if (!analysis) {
        summaryDiv.innerText = 'Load an enriched influence file to render semantic message influence.';
        return;
    }

    const messageCount = (analysis.messages || []).length;
    const graphInfo = analysis.influence_graph || {};
    summaryDiv.innerText = `Messages: ${messageCount} | Influence edges: ${graphInfo.edge_count || 0} | Filtered edges: ${graphInfo.filtered_edge_count || 0}`;
}

function drawDegreeDistribution() {
    const degrees = simData.nodes.map(node => node.degree);
    const maxDegree = Math.max(...degrees, 0);
    const bins = new Array(maxDegree + 1).fill(0);
    degrees.forEach(degree => {
        bins[degree] += 1;
    });

    const ctx = document.getElementById('degreeChart').getContext('2d');
    if (degreeChart) {
        degreeChart.destroy();
    }

    degreeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Array.from({ length: maxDegree + 1 }, (_, index) => index),
            datasets: [{
                label: 'Number of Nodes',
                data: bins,
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1,
            }],
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: 'Degree' } },
                y: { title: { display: true, text: 'Count' }, type: 'logarithmic' },
            },
        },
    });
}

function drawPowerlawDistribution() {
    const degrees = simData.nodes.map(node => node.degree);
    const maxDegree = Math.max(...degrees, 0);
    const bins = new Array(maxDegree + 1).fill(0);
    degrees.forEach(degree => {
        bins[degree] += 1;
    });

    const scatterData = [];
    for (let degree = 1; degree <= maxDegree; degree += 1) {
        if (bins[degree] > 0) {
            scatterData.push({ x: degree, y: bins[degree] });
        }
    }

    const ctx = document.getElementById('powerlawChart').getContext('2d');
    if (powerlawChart) {
        powerlawChart.destroy();
    }

    powerlawChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Degree Distribution',
                data: scatterData,
                backgroundColor: 'rgba(255, 99, 132, 0.5)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1,
            }],
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: { display: true, text: 'Degree (log)' },
                    type: 'logarithmic',
                },
                y: {
                    title: { display: true, text: 'Count (log)' },
                    type: 'logarithmic',
                },
            },
        },
    });
}

function drawSurveyChart() {
    if (simData.survey_results.length > 0) {
        document.getElementById('surveyQuestion').innerText = 'Question: ' + simData.survey_results[0].question;
    } else {
        document.getElementById('surveyQuestion').innerText = 'No survey results available.';
    }

    const ctx = document.getElementById('surveyChart').getContext('2d');
    if (surveyChart) {
        surveyChart.destroy();
    }

    const steps = simData.survey_results.map(result => result.step);
    const yAxisMode = document.querySelector('input[name="surveyYAxis"]:checked').value;
    const allAnswers = new Set();
    simData.survey_results.forEach(result => {
        Object.values(result.results).forEach(answer => allAnswers.add(answer));
    });

    const colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#8BC34A'];
    const datasets = Array.from(allAnswers).map((answer, index) => {
        const data = simData.survey_results.map(result => {
            let count = 0;
            const totalResponses = Object.keys(result.results).length;
            Object.values(result.results).forEach(value => {
                if (value === answer) {
                    count += 1;
                }
            });

            const yValue = yAxisMode === 'percentage' && totalResponses > 0
                ? (count / totalResponses) * 100
                : count;
            return { x: result.step, y: yValue };
        });
        return {
            label: answer,
            data,
            borderColor: colors[index % colors.length],
            fill: false,
            tension: 0.1,
        };
    });

    const annotations = {};
    simData.news_posts.forEach((post, index) => {
        annotations[`line${index}`] = {
            type: 'line',
            xMin: post.step,
            xMax: post.step,
            borderColor: 'rgba(255, 99, 132, 0.5)',
            borderWidth: 2,
            borderDash: [5, 5],
            label: {
                content: 'News',
                display: true,
                position: 'start',
            },
        };
    });

    const yAxisMax = yAxisMode === 'percentage'
        ? 100
        : (simData.survey_results.length > 0 ? Object.keys(simData.survey_results[0].results).length : undefined);

    surveyChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            plugins: {
                annotation: {
                    annotations: document.getElementById('toggleNewsLines').checked ? annotations : {},
                },
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'Step' },
                    ticks: {
                        stepSize: steps.length > 1 ? steps[1] - steps[0] : 100,
                    },
                },
                y: {
                    title: { display: true, text: yAxisMode === 'percentage' ? 'Percentage (%)' : 'Count' },
                    min: 0,
                    max: yAxisMax,
                },
            },
        },
    });
}

function updateSurveyChart() {
    drawSurveyChart();
}

function computeOpinionChangesCounts(nodeCount) {
    const changesCount = new Array(nodeCount).fill(0);
    for (let index = 1; index < simData.survey_results.length; index += 1) {
        const previousSurvey = simData.survey_results[index - 1].results;
        const currentSurvey = simData.survey_results[index].results;
        simData.nodes.forEach((node, nodeIndex) => {
            if (node.type !== 'NewsSource' && previousSurvey[node.name] && currentSurvey[node.name] && previousSurvey[node.name] !== currentSurvey[node.name]) {
                changesCount[nodeIndex] += 1;
            }
        });
    }
    return changesCount;
}

function drawOpinionChangesDistribution() {
    const changesCount = computeOpinionChangesCounts(simData.nodes.length);
    const maxChanges = Math.max(...changesCount, 0);
    const bins = new Array(maxChanges + 1).fill(0);
    changesCount.forEach((count, index) => {
        if (simData.nodes[index].type !== 'NewsSource') {
            bins[count] += 1;
        }
    });

    const ctx = document.getElementById('opinionChangesChart').getContext('2d');
    if (opinionChangesChart) {
        opinionChangesChart.destroy();
    }

    opinionChangesChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Array.from({ length: maxChanges + 1 }, (_, index) => index),
            datasets: [{
                label: 'Number of Nodes',
                data: bins,
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1,
            }],
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: 'Number of Opinion Changes' } },
                y: { title: { display: true, text: 'Count' } },
            },
        },
    });
}

function drawSemanticDriftChart() {
    const analysis = getInfluenceAnalysis();
    const driftSeries = analysis ? (analysis.discourse_drift.global || []) : [];
    const ctx = document.getElementById('semanticDriftChart').getContext('2d');

    if (semanticDriftChart) {
        semanticDriftChart.destroy();
    }

    semanticDriftChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: driftSeries.map(point => point.window_start),
            datasets: [{
                label: 'Semantic Drift',
                data: driftSeries.map(point => point.drift),
                borderColor: '#0f766e',
                backgroundColor: 'rgba(15, 118, 110, 0.15)',
                fill: true,
                tension: 0.2,
            }],
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: 'Window Start Step' } },
                y: { title: { display: true, text: 'Mean Embedding Shift' }, min: 0 },
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        afterLabel(context) {
                            const point = driftSeries[context.dataIndex];
                            return point ? `Messages: ${point.message_count}` : '';
                        },
                    },
                },
            },
        },
    });
}

function drawImpactTable() {
    const analysis = getInfluenceAnalysis();
    const tbody = document.querySelector('#impactTable tbody');
    tbody.innerHTML = '';

    if (!analysis) {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="5">No influence analysis loaded.</td>';
        tbody.appendChild(row);
        return;
    }

    const messageMap = getInfluenceMessageMap();
    (analysis.impact_scores || []).slice(0, 100).forEach((entry, index) => {
        const message = messageMap.get(entry.message_id);
        if (!message) {
            return;
        }

        const row = document.createElement('tr');
        if (selectedMessageId === message.id) {
            row.classList.add('impact-row-selected');
        }
        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${escapeHtml(message.author)}</td>
            <td>${message.timestamp}</td>
            <td>${formatNumber(message.impact_score)}</td>
            <td>${formatNumber(message.drift_score)}</td>
        `;
        row.title = truncateText(message.content, 180);
        row.addEventListener('click', () => selectMessage(message.id));
        tbody.appendChild(row);
    });
}

function drawMessageReach() {
    const tbody = document.querySelector('#messageReachTable tbody');
    tbody.innerHTML = '';

    const threadViews = {};
    simData.observations.forEach(observation => {
        if (observation.step <= currentStep) {
            threadViews[observation.thread_id] = (threadViews[observation.thread_id] || 0) + 1;
        }
    });

    const threadInfo = [];
    simData.threads.forEach(thread => {
        if (thread.messages && thread.messages.length > 0) {
            threadInfo.push({
                id: thread.id,
                author: thread.messages[0].role,
                content: truncateText(cleanMessageContent(thread.messages[0].content), 50),
                views: threadViews[thread.id] || 0,
            });
        }
    });

    threadInfo.sort((left, right) => right.views - left.views);
    threadInfo.slice(0, 50).forEach(info => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${info.id}</td>
            <td>${escapeHtml(info.author)}</td>
            <td title="${escapeHtml(info.content)}" class="message-cell">${escapeHtml(info.content)}</td>
            <td>${info.views}</td>
        `;
        row.querySelector('.message-cell').addEventListener('click', () => showThreadDetails(info.id));
        tbody.appendChild(row);
    });
}

function showThreadDetails(threadId) {
    const thread = simData.threads.find(item => item.id === threadId);
    if (!thread) {
        return;
    }

    let totalThreadViews = 0;
    simData.observations.forEach(observation => {
        if (observation.thread_id === threadId && observation.step <= currentStep) {
            totalThreadViews += 1;
        }
    });

    let content = `<h3>Thread ${threadId} (Seen ${totalThreadViews} times)</h3><div style="max-height: 400px; overflow-y: auto;">`;
    thread.messages.forEach(message => {
        if (message.step <= currentStep) {
            let messageViews = 0;
            simData.observations.forEach(observation => {
                if (observation.thread_id === threadId && observation.step >= message.step && observation.step <= currentStep) {
                    messageViews += 1;
                }
            });
            content += `<div class="message-item"><strong>${escapeHtml(message.role)}</strong> (Step ${message.step || 'N/A'}) [Seen ${messageViews} times]: ${escapeHtml(cleanMessageContent(message.content))}</div>`;
        }
    });
    content += '</div>';

    const infoDiv = document.getElementById('node-info');
    infoDiv.innerHTML = content;
    document.getElementById('node-threads').innerHTML = '';
    document.getElementById('node-messages').innerHTML = '';
}

function initGraph() {
    const container = document.getElementById('graph-container');
    container.innerHTML = '';
    const width = container.clientWidth;
    const height = container.clientHeight || 500;

    const svg = d3.select('#graph-container').append('svg')
        .attr('width', width)
        .attr('height', height)
        .call(d3.zoom().on('zoom', event => {
            group.attr('transform', event.transform);
        }));

    svg.on('click', event => {
        if (event.target.tagName === 'svg') {
            selectNode(null);
        }
    });

    const group = svg.append('g');

    simulationGraph = d3.forceSimulation(simData.nodes)
        .force('link', d3.forceLink(simData.edges).id(node => node.id).distance(30))
        .force('charge', d3.forceManyBody().strength(-50))
        .force('center', d3.forceCenter(width / 2, height / 2));

    const link = group.append('g')
        .attr('class', 'links')
        .selectAll('line')
        .data(simData.edges)
        .enter()
        .append('line')
        .attr('class', 'link');

    const node = group.append('g')
        .attr('class', 'nodes')
        .selectAll('circle')
        .data(simData.nodes)
        .enter()
        .append('circle')
        .attr('class', 'node')
        .attr('r', datum => datum.type === 'NewsSource' ? 8 : 5)
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended))
        .on('click', (event, datum) => {
            event.stopPropagation();
            selectNode(datum);
        });

    node.append('title').text(datum => datum.name);

    simulationGraph.on('tick', () => {
        link
            .attr('x1', datum => datum.source.x)
            .attr('y1', datum => datum.source.y)
            .attr('x2', datum => datum.target.x)
            .attr('y2', datum => datum.target.y);

        node
            .attr('cx', datum => datum.x)
            .attr('cy', datum => datum.y);
    });

    function dragstarted(event, datum) {
        if (!event.active) {
            simulationGraph.alphaTarget(0.3).restart();
        }
        datum.fx = datum.x;
        datum.fy = datum.y;
    }

    function dragged(event, datum) {
        datum.fx = event.x;
        datum.fy = event.y;
    }

    function dragended(event, datum) {
        if (!event.active) {
            simulationGraph.alphaTarget(0);
        }
        datum.fx = null;
        datum.fy = null;
    }
}

function updateGraph() {
    if (!simData || !simulationGraph) {
        return;
    }

    const highlightChanged = document.getElementById('toggleChangedMinds').checked;
    const colorByOpinionChange = document.getElementById('toggleOpinionChangeColor').checked;
    const colorByLatestVote = document.getElementById('toggleLatestVoteColor').checked;
    const sizeByMessages = document.getElementById('toggleNodeSize').checked;
    const sizeByViews = document.getElementById('toggleNodeSizeViews').checked;
    const currentSurvey = simData.survey_results.find(result => result.step === currentStep);
    const currentIndex = surveySteps.indexOf(currentStep);
    const previousStep = currentIndex > 0 ? surveySteps[currentIndex - 1] : null;
    const previousSurvey = previousStep !== null
        ? simData.survey_results.find(result => result.step === previousStep)
        : null;

    const changesCount = colorByOpinionChange ? computeOpinionChangesCounts(simData.nodes.length) : new Array(simData.nodes.length).fill(0);
    const maxChanges = Math.max(...changesCount, 1);

    const messagesWritten = new Array(simData.nodes.length).fill(0);
    if (sizeByMessages) {
        simData.threads.forEach(thread => {
            (thread.messages || []).forEach(message => {
                if (message.step <= currentStep) {
                    const nodeIndex = simData.nodes.findIndex(node => node.name === message.role);
                    if (nodeIndex !== -1) {
                        messagesWritten[nodeIndex] += 1;
                    }
                }
            });
        });
    }
    const maxMessages = Math.max(...messagesWritten, 1);

    const cumulativeViews = new Array(simData.nodes.length).fill(0);
    if (sizeByViews) {
        const threadMap = new Map();
        simData.threads.forEach(thread => threadMap.set(thread.id, thread));
        const nodeIndexMap = new Map();
        simData.nodes.forEach((node, index) => nodeIndexMap.set(node.name, index));

        simData.observations.forEach(observation => {
            if (observation.step <= currentStep) {
                const thread = threadMap.get(observation.thread_id);
                (thread?.messages || []).forEach(message => {
                    if (message.step <= observation.step) {
                        const nodeIndex = nodeIndexMap.get(message.role);
                        if (nodeIndex !== undefined) {
                            cumulativeViews[nodeIndex] += 1;
                        }
                    }
                });
            }
        });
    }
    const maxViews = Math.max(...cumulativeViews, 1);

    const allAnswers = Array.from(new Set(simData.survey_results.flatMap(result => Object.values(result.results))));
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10).domain(allAnswers);

    d3.selectAll('.node')
        .attr('r', (datum, index) => {
            if (datum.type === 'NewsSource') {
                return 8;
            }
            if (sizeByViews) {
                return 3 + (cumulativeViews[index] / maxViews) * 15;
            }
            if (sizeByMessages) {
                return 3 + (messagesWritten[index] / maxMessages) * 15;
            }
            return 5;
        })
        .attr('fill', (datum, index) => {
            if (datum.type === 'NewsSource') {
                return '#000';
            }

            let nodeColor = colorByLatestVote && currentSurvey && currentSurvey.results[datum.name]
                ? d3.color(colorScale(currentSurvey.results[datum.name]))
                : d3.color('#1f77b4');

            if (colorByOpinionChange) {
                const changeRatio = changesCount[index] / maxChanges;
                const offset = changeRatio * 3 - 1;
                nodeColor = offset > 0 ? nodeColor.darker(offset) : nodeColor.brighter(-offset);
            }

            if (highlightChanged) {
                if (!previousSurvey || !currentSurvey || !previousSurvey.results[datum.name] || !currentSurvey.results[datum.name]) {
                    nodeColor = d3.color('#eee');
                } else if (previousSurvey.results[datum.name] === currentSurvey.results[datum.name]) {
                    nodeColor = d3.color('#eee');
                }
            }

            return nodeColor.formatHex();
        })
        .attr('stroke', datum => datum === selectedNode ? '#000' : '#fff')
        .attr('stroke-width', datum => datum === selectedNode ? 3 : 1.5)
        .attr('opacity', datum => selectedNode && datum !== selectedNode ? 0.3 : 1);

    d3.selectAll('.link')
        .attr('stroke-opacity', datum => {
            if (!selectedNode) {
                return 0.6;
            }
            return datum.source === selectedNode || datum.target === selectedNode ? 1 : 0.1;
        });

    updateNodeDetails(selectedNode);
}

function updateInfluenceGraph() {
    const analysis = getInfluenceAnalysis();
    const container = document.getElementById('message-graph-container');
    container.innerHTML = '';

    if (!analysis) {
        container.innerHTML = '<div class="empty-state">No influence analysis loaded.</div>';
        drawImpactTable();
        return;
    }

    const messages = analysis.messages || [];
    const threshold = getInfluenceThreshold();
    const nodeLimit = parseInt(document.getElementById('influenceNodeLimit').value, 10);
    const rankedMessages = messages.slice().sort((left, right) => right.impact_score - left.impact_score);
    const selectedMessage = selectedMessageId !== null ? messages.find(message => message.id === selectedMessageId) : null;
    const nodeIds = new Set(rankedMessages.slice(0, nodeLimit).map(message => message.id));

    if (selectedMessage) {
        nodeIds.add(selectedMessage.id);
        (selectedMessage.top_parents || []).forEach(edge => nodeIds.add(edge.source));
        (selectedMessage.top_children || []).forEach(edge => nodeIds.add(edge.target));
    }

    const nodes = messages.filter(message => nodeIds.has(message.id)).map(message => ({ ...message }));
    const nodeMap = new Map(nodes.map(node => [node.id, node]));
    const edges = (analysis.influence_graph.edges || [])
        .filter(edge => edge.probability >= threshold && nodeMap.has(edge.source) && nodeMap.has(edge.target))
        .map(edge => ({ ...edge }));

    if (nodes.length === 0) {
        container.innerHTML = '<div class="empty-state">No influence nodes match the current filters.</div>';
        drawImpactTable();
        return;
    }

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 500;
    const maxImpact = Math.max(...nodes.map(node => node.impact_score), 1e-6);
    const maxEdgeProbability = Math.max(...edges.map(edge => edge.probability), threshold, 0.01);
    const radiusScale = d3.scaleSqrt().domain([0, maxImpact]).range([4, 18]);
    const driftExtent = d3.extent(nodes.map(node => node.drift_score || 0));
    const colorScale = d3.scaleLinear()
        .domain([driftExtent[0] ?? -1, 0, driftExtent[1] ?? 1])
        .range(['#c2410c', '#cbd5e1', '#0f766e']);

    const svg = d3.select('#message-graph-container').append('svg')
        .attr('width', width)
        .attr('height', height)
        .call(d3.zoom().on('zoom', event => {
            group.attr('transform', event.transform);
        }));

    const tooltip = d3.select('#message-graph-container')
        .append('div')
        .attr('class', 'tooltip')
        .style('opacity', 0);

    const group = svg.append('g');

    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(edges).id(node => node.id).distance(60))
        .force('charge', d3.forceManyBody().strength(-120))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(node => radiusScale(node.impact_score) + 4));

    const link = group.append('g')
        .selectAll('line')
        .data(edges)
        .enter()
        .append('line')
        .attr('class', 'message-link')
        .attr('stroke-width', edge => 1 + (edge.probability / maxEdgeProbability) * 4);

    const node = group.append('g')
        .selectAll('circle')
        .data(nodes)
        .enter()
        .append('circle')
        .attr('class', 'message-node')
        .attr('r', message => radiusScale(message.impact_score))
        .attr('fill', message => colorScale(message.drift_score || 0))
        .attr('stroke', message => message.id === selectedMessageId ? '#111827' : '#ffffff')
        .attr('stroke-width', message => message.id === selectedMessageId ? 3 : 1.2)
        .on('mouseover', function(event, message) {
            tooltip.style('opacity', 0.95)
                .html([
                    `<strong>${escapeHtml(message.author)}</strong>`,
                    `Step ${message.timestamp}`,
                    `Impact: ${formatNumber(message.impact_score)}`,
                    `Drift: ${formatNumber(message.drift_score)}`,
                    escapeHtml(truncateText(message.content, 140)),
                ].join('<br>'))
                .style('left', `${event.offsetX + 12}px`)
                .style('top', `${event.offsetY + 12}px`);
        })
        .on('mousemove', function(event) {
            tooltip.style('left', `${event.offsetX + 12}px`).style('top', `${event.offsetY + 12}px`);
        })
        .on('mouseout', function() {
            tooltip.style('opacity', 0);
        })
        .on('click', function(event, message) {
            event.stopPropagation();
            selectMessage(message.id);
        })
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));

    node.append('title').text(message => truncateText(message.content, 160));

    simulation.on('tick', () => {
        link
            .attr('x1', edge => edge.source.x)
            .attr('y1', edge => edge.source.y)
            .attr('x2', edge => edge.target.x)
            .attr('y2', edge => edge.target.y);

        node
            .attr('cx', message => message.x)
            .attr('cy', message => message.y);
    });

    svg.on('click', function(event) {
        if (event.target.tagName === 'svg') {
            selectMessage(null);
        }
    });

    function dragstarted(event, datum) {
        if (!event.active) {
            simulation.alphaTarget(0.3).restart();
        }
        datum.fx = datum.x;
        datum.fy = datum.y;
    }

    function dragged(event, datum) {
        datum.fx = event.x;
        datum.fy = event.y;
    }

    function dragended(event, datum) {
        if (!event.active) {
            simulation.alphaTarget(0);
        }
        datum.fx = null;
        datum.fy = null;
    }

    drawImpactTable();
}

function selectNode(node) {
    selectedNode = node;
    updateGraph();
}

function selectMessage(messageId) {
    selectedMessageId = messageId;
    updateInfluenceGraph();
    updateMessageDetails(messageId);
    updateCascadeTree(messageId);
}

function updateNodeDetails(node) {
    const infoDiv = document.getElementById('node-info');
    if (!node) {
        infoDiv.innerHTML = 'Select a node to see details.';
        document.getElementById('node-threads').innerHTML = '';
        document.getElementById('node-messages').innerHTML = '';
        return;
    }

    infoDiv.innerHTML = `
        <strong>Name:</strong> ${escapeHtml(node.name)}<br>
        <strong>Type:</strong> ${escapeHtml(node.type)}<br>
        <strong>Model ID:</strong> ${node.model_id ?? 'N/A'}<br>
        <strong>Degree:</strong> ${node.degree}
    `;

    const seenThreads = new Set();
    simData.observations.forEach(observation => {
        if (observation.entity_name === node.name && observation.step <= currentStep) {
            seenThreads.add(observation.thread_id);
        }
    });

    const threadsDiv = document.getElementById('node-threads');
    threadsDiv.innerHTML = `<h3>Threads Seen (${seenThreads.size})</h3>`;
    const threadsList = document.createElement('div');
    threadsList.style.maxHeight = '300px';
    threadsList.style.overflowY = 'auto';
    Array.from(seenThreads).forEach(threadId => {
        const div = document.createElement('div');
        div.className = 'thread-item clickable';
        div.innerText = `Thread ${threadId}`;
        div.onclick = () => showThreadDetails(threadId);
        threadsList.appendChild(div);
    });
    threadsDiv.appendChild(threadsList);

    const writtenMessages = [];
    simData.threads.forEach(thread => {
        (thread.messages || []).forEach(message => {
            if (message.role === node.name && message.step <= currentStep) {
                let messageViews = 0;
                simData.observations.forEach(observation => {
                    if (observation.thread_id === thread.id && observation.step >= message.step && observation.step <= currentStep) {
                        messageViews += 1;
                    }
                });
                writtenMessages.push({
                    thread_id: thread.id,
                    content: cleanMessageContent(message.content),
                    step: message.step,
                    views: messageViews,
                });
            }
        });
    });

    const messagesDiv = document.getElementById('node-messages');
    messagesDiv.innerHTML = `<h3>Messages Written (${writtenMessages.length})</h3>`;
    const messagesList = document.createElement('div');
    messagesList.style.maxHeight = '300px';
    messagesList.style.overflowY = 'auto';

    let totalViews = 0;
    writtenMessages.sort((left, right) => right.step - left.step).forEach(message => {
        totalViews += message.views;
        const div = document.createElement('div');
        div.className = 'message-item clickable';
        div.innerHTML = `<strong>Step ${message.step} (Thread ${message.thread_id}) [Seen ${message.views} times]:</strong> ${escapeHtml(message.content)}`;
        div.onclick = () => showThreadDetails(message.thread_id);
        messagesList.appendChild(div);
    });
    messagesDiv.appendChild(messagesList);

    infoDiv.innerHTML += `<br><strong>Total Views:</strong> ${totalViews}`;
}

function updateMessageDetails(messageId) {
    const analysis = getInfluenceAnalysis();
    const infoDiv = document.getElementById('message-info');
    const parentsDiv = document.getElementById('message-parents');
    const childrenDiv = document.getElementById('message-children');

    if (!analysis || messageId === null) {
        infoDiv.innerHTML = 'Select a message in the influence graph or ranking table.';
        parentsDiv.innerHTML = '';
        childrenDiv.innerHTML = '';
        drawImpactTable();
        return;
    }

    const message = getInfluenceMessages().find(item => item.id === messageId);
    if (!message) {
        infoDiv.innerHTML = 'Selected message not found.';
        parentsDiv.innerHTML = '';
        childrenDiv.innerHTML = '';
        drawImpactTable();
        return;
    }

    infoDiv.innerHTML = `
        <div class="metric-card"><strong>Author:</strong> ${escapeHtml(message.author)}</div>
        <div class="metric-card"><strong>Step:</strong> ${message.timestamp}</div>
        <div class="metric-card"><strong>Impact Score:</strong> ${formatNumber(message.impact_score)}</div>
        <div class="metric-card"><strong>Drift Score:</strong> ${formatNumber(message.drift_score)}</div>
        <div class="metric-card"><strong>Viewers:</strong> ${message.viewer_count}</div>
        <div class="metric-card"><strong>Text:</strong> ${escapeHtml(message.content)}</div>
    `;

    parentsDiv.innerHTML = buildMessageEdgeList('Top Parent Messages', message.top_parents || []);
    childrenDiv.innerHTML = buildMessageEdgeList('Top Influenced Messages', message.top_children || []);
    drawImpactTable();
}

function buildMessageEdgeList(title, edges) {
    if (!edges.length) {
        return `<h3>${title}</h3><div class="empty-state">No edges available.</div>`;
    }

    const messageMap = getInfluenceMessageMap();
    const items = edges.slice(0, 5).map(edge => {
        const relatedId = edge.source === selectedMessageId ? edge.target : edge.source;
        const relatedMessage = messageMap.get(relatedId);
        const label = relatedMessage
            ? `${escapeHtml(relatedMessage.author)} @ ${relatedMessage.timestamp}`
            : `Message ${relatedId}`;
        const content = relatedMessage ? escapeHtml(truncateText(relatedMessage.content, 90)) : '';
        return `
            <div class="metric-card clickable" onclick="selectMessage(${relatedId})">
                <strong>${label}</strong><br>
                P=${formatNumber(edge.probability)} | sim=${formatNumber(edge.similarity)} | dt=${formatNumber(edge.time_delta || 0)}<br>
                ${content}
            </div>
        `;
    }).join('');

    return `<h3>${title}</h3>${items}`;
}

function updateCascadeTree(messageId) {
    const analysis = getInfluenceAnalysis();
    const summaryDiv = document.getElementById('cascade-summary');
    const container = document.getElementById('cascade-tree-container');
    container.innerHTML = '';

    if (!analysis || messageId === null) {
        summaryDiv.innerHTML = 'Select a message to inspect its cascade.';
        return;
    }

    const cascades = analysis.cascades || {};
    const cascadeEntry = (cascades.by_message || {})[String(messageId)];
    const message = getInfluenceMessages().find(item => item.id === messageId);
    if (!cascadeEntry || !message) {
        summaryDiv.innerHTML = 'Cascade data unavailable for the selected message.';
        return;
    }

    summaryDiv.innerHTML = `
        <strong>Root Author:</strong> ${escapeHtml(message.author)}<br>
        <strong>Cascade Size:</strong> ${cascadeEntry.cascade_size}<br>
        <strong>Direct Children:</strong> ${cascadeEntry.direct_children.length}<br>
        <strong>Tree Probability Mass:</strong> ${formatNumber(cascadeEntry.tree_probability_mass)}
    `;

    const treeEdges = (cascades.best_parent_tree_edges || []).filter(edge => edge.probability >= getInfluenceThreshold());
    const childrenBySource = new Map();
    treeEdges.forEach(edge => {
        if (!childrenBySource.has(edge.source)) {
            childrenBySource.set(edge.source, []);
        }
        childrenBySource.get(edge.source).push(edge.target);
    });

    function buildHierarchy(rootId, visited = new Set()) {
        visited.add(rootId);
        const rootMessage = getInfluenceMessageMap().get(rootId);
        const children = (childrenBySource.get(rootId) || [])
            .filter(childId => !visited.has(childId))
            .map(childId => buildHierarchy(childId, new Set(visited)));
        return {
            name: rootMessage ? `${rootMessage.author} @ ${rootMessage.timestamp}` : `Message ${rootId}`,
            id: rootId,
            children,
        };
    }

    const hierarchy = d3.hierarchy(buildHierarchy(messageId));
    const width = container.clientWidth || 300;
    const height = Math.max(280, hierarchy.descendants().length * 24);
    const treeLayout = d3.tree().size([height - 30, width - 80]);
    treeLayout(hierarchy);

    const svg = d3.select('#cascade-tree-container').append('svg')
        .attr('width', width)
        .attr('height', height);
    const group = svg.append('g').attr('transform', 'translate(40,15)');

    group.selectAll('path')
        .data(hierarchy.links())
        .enter()
        .append('path')
        .attr('class', 'cascade-link')
        .attr('d', d3.linkHorizontal().x(link => link.y).y(link => link.x));

    const nodes = group.selectAll('g')
        .data(hierarchy.descendants())
        .enter()
        .append('g')
        .attr('transform', node => `translate(${node.y},${node.x})`)
        .style('cursor', 'pointer')
        .on('click', (_, node) => selectMessage(node.data.id));

    nodes.append('circle')
        .attr('class', 'cascade-node')
        .attr('r', node => node.data.id === messageId ? 7 : 5)
        .attr('fill', node => node.data.id === messageId ? '#1d4ed8' : '#0f766e');

    nodes.append('text')
        .attr('dy', 4)
        .attr('x', node => node.children ? -8 : 8)
        .style('text-anchor', node => node.children ? 'end' : 'start')
        .style('font-size', '11px')
        .text(node => truncateText(node.data.name, 28));
}

function focusPanel(panelId) {
    const panels = document.querySelectorAll('.focusable');
    panels.forEach(panel => {
        if (panel.id === panelId) {
            panel.classList.add('focused');
            document.getElementById('center-panel').appendChild(panel);
        } else {
            panel.classList.remove('focused');
            document.getElementById('left-panel').appendChild(panel);
        }
    });

    if (degreeChart) {
        degreeChart.resize();
    }
    if (powerlawChart) {
        powerlawChart.resize();
    }
    if (surveyChart) {
        surveyChart.resize();
    }
    if (opinionChangesChart) {
        opinionChangesChart.resize();
    }
    if (semanticDriftChart) {
        semanticDriftChart.resize();
    }

    if (panelId === 'graph-panel' && simulationGraph) {
        const container = document.getElementById('graph-container');
        const width = container.clientWidth;
        const height = container.clientHeight || 500;

        d3.select('#graph-container svg')
            .attr('width', width)
            .attr('height', height);

        simulationGraph.force('center', d3.forceCenter(width / 2, height / 2));
        simulationGraph.alpha(0.3).restart();
    }

    if (panelId === 'influence-panel') {
        updateInfluenceGraph();
    }
}

function cleanMessageContent(text) {
    return String(text || '')
        .replaceAll('<|im_end|>', ' ')
        .replaceAll('<URL>', 'URL')
        .replace(/\s+/g, ' ')
        .trim();
}

function truncateText(text, maxLength) {
    const value = String(text || '');
    return value.length > maxLength ? `${value.slice(0, maxLength - 3)}...` : value;
}

function escapeHtml(value) {
    return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

function formatNumber(value) {
    const numericValue = Number(value || 0);
    if (!Number.isFinite(numericValue)) {
        return '0';
    }
    if (Math.abs(numericValue) >= 1000) {
        return numericValue.toFixed(0);
    }
    if (Math.abs(numericValue) >= 10) {
        return numericValue.toFixed(2);
    }
    return numericValue.toFixed(3);
}
