let simData = null;
let degreeChart = null;
let surveyChart = null;
let simulationGraph = null;
let currentStep = 0;
let maxStep = 0;
let surveySteps = [];
let selectedNode = null;

document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            simData = JSON.parse(e.target.result);
            processData();
            initVisualizations();
        } catch (err) {
            console.error("Error parsing JSON:", err);
            alert("Invalid JSON file.");
        }
    };
    reader.readAsText(file);
});

function processData() {
    // Calculate degrees
    const degrees = new Array(simData.nodes.length).fill(0);
    simData.edges.forEach(edge => {
        degrees[edge.source]++;
        degrees[edge.target]++;
    });
    simData.nodes.forEach((node, i) => {
        node.degree = degrees[i];
    });

    // Find max step
    maxStep = 0;
    if (simData.survey_results.length > 0) {
        maxStep = Math.max(maxStep, ...simData.survey_results.map(s => s.step));
    }
    if (simData.observations.length > 0) {
        maxStep = Math.max(maxStep, ...simData.observations.map(o => o.step));
    }
    
    surveySteps = simData.survey_results.map(s => s.step).sort((a, b) => a - b);

    // Setup slider
    const slider = document.getElementById('timeSlider');
    slider.max = surveySteps.length > 0 ? surveySteps.length - 1 : 0;
    slider.value = 0;
    currentStep = surveySteps.length > 0 ? surveySteps[0] : 0;
    document.getElementById('timeValue').innerText = currentStep;

    slider.addEventListener('input', function() {
        const idx = parseInt(this.value);
        currentStep = surveySteps[idx];
        document.getElementById('timeValue').innerText = currentStep;
        updateGraph();
    });

    document.getElementById('toggleChangedMinds').addEventListener('change', updateGraph);
    document.getElementById('toggleNewsLines').addEventListener('change', updateSurveyChart);
}

function initVisualizations() {
    drawDegreeDistribution();
    drawSurveyChart();
    drawMessageReach();
    initGraph();
    updateGraph();
}

function drawDegreeDistribution() {
    const degrees = simData.nodes.map(n => n.degree);
    const maxDegree = Math.max(...degrees);
    const bins = new Array(maxDegree + 1).fill(0);
    degrees.forEach(d => bins[d]++);

    const ctx = document.getElementById('degreeChart').getContext('2d');
    if (degreeChart) degreeChart.destroy();

    degreeChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Array.from({length: maxDegree + 1}, (_, i) => i),
            datasets: [{
                label: 'Number of Nodes',
                data: bins,
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: 'Degree' } },
                y: { title: { display: true, text: 'Count' }, type: 'logarithmic' }
            }
        }
    });
}

function drawSurveyChart() {
    const ctx = document.getElementById('surveyChart').getContext('2d');
    if (surveyChart) surveyChart.destroy();

    const steps = simData.survey_results.map(s => s.step);
    
    // Get all unique answers
    const allAnswers = new Set();
    simData.survey_results.forEach(s => {
        Object.values(s.results).forEach(ans => allAnswers.add(ans));
    });
    
    const datasets = Array.from(allAnswers).map((ans, i) => {
        const colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40'];
        const data = simData.survey_results.map(s => {
            let count = 0;
            Object.values(s.results).forEach(val => {
                if (val === ans) count++;
            });
            return { x: s.step, y: count };
        });
        return {
            label: ans,
            data: data,
            borderColor: colors[i % colors.length],
            fill: false,
            tension: 0.1
        };
    });

    const annotations = {};
    simData.news_posts.forEach((post, i) => {
        annotations[`line${i}`] = {
            type: 'line',
            xMin: post.step,
            xMax: post.step,
            borderColor: 'rgba(255, 99, 132, 0.5)',
            borderWidth: 2,
            borderDash: [5, 5],
            label: {
                content: 'News',
                display: true,
                position: 'start'
            }
        };
    });

    surveyChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            plugins: {
                annotation: {
                    annotations: document.getElementById('toggleNewsLines').checked ? annotations : {}
                }
            },
            scales: {
                x: { 
                    type: 'linear',
                    title: { display: true, text: 'Step' },
                    ticks: {
                        stepSize: steps.length > 1 ? steps[1] - steps[0] : 100
                    }
                },
                y: { title: { display: true, text: 'Count' } }
            }
        }
    });
}

function updateSurveyChart() {
    if (!surveyChart) return;
    const annotations = {};
    if (document.getElementById('toggleNewsLines').checked) {
        simData.news_posts.forEach((post, i) => {
            annotations[`line${i}`] = {
                type: 'line',
                xMin: post.step,
                xMax: post.step,
                borderColor: 'rgba(255, 99, 132, 0.5)',
                borderWidth: 2,
                borderDash: [5, 5]
            };
        });
    }
    surveyChart.options.plugins.annotation.annotations = annotations;
    surveyChart.update();
}

function drawMessageReach() {
    const tbody = document.querySelector('#messageReachTable tbody');
    tbody.innerHTML = '';

    // Calculate views per thread
    const threadViews = {};
    simData.observations.forEach(obs => {
        threadViews[obs.thread_id] = (threadViews[obs.thread_id] || 0) + 1;
    });

    // Get first message of each thread
    const threadInfo = [];
    simData.threads.forEach(t => {
        if (t.messages && t.messages.length > 0) {
            threadInfo.push({
                id: t.id,
                author: t.messages[0].role,
                content: t.messages[0].content.substring(0, 50) + '...',
                views: threadViews[t.id] || 0
            });
        }
    });

    threadInfo.sort((a, b) => b.views - a.views);

    threadInfo.slice(0, 50).forEach(info => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${info.id}</td>
            <td>${info.author}</td>
            <td title="${info.content}">${info.content}</td>
            <td>${info.views}</td>
        `;
        tbody.appendChild(tr);
    });
}

function initGraph() {
    const container = document.getElementById('graph-container');
    container.innerHTML = '';
    const width = container.clientWidth;
    const height = container.clientHeight || 500;

    const svg = d3.select('#graph-container').append('svg')
        .attr('width', width)
        .attr('height', height)
        .call(d3.zoom().on('zoom', (event) => {
            g.attr('transform', event.transform);
        }));

    const g = svg.append('g');

    simulationGraph = d3.forceSimulation(simData.nodes)
        .force('link', d3.forceLink(simData.edges).id(d => d.id).distance(30))
        .force('charge', d3.forceManyBody().strength(-50))
        .force('center', d3.forceCenter(width / 2, height / 2));

    const link = g.append('g')
        .attr('class', 'links')
        .selectAll('line')
        .data(simData.edges)
        .enter().append('line')
        .attr('class', 'link');

    const node = g.append('g')
        .attr('class', 'nodes')
        .selectAll('circle')
        .data(simData.nodes)
        .enter().append('circle')
        .attr('class', 'node')
        .attr('r', d => d.type === 'NewsSource' ? 8 : 5)
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended))
        .on('click', (event, d) => selectNode(d));

    node.append('title')
        .text(d => d.name);

    simulationGraph.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
    });

    function dragstarted(event, d) {
        if (!event.active) simulationGraph.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    function dragended(event, d) {
        if (!event.active) simulationGraph.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

function updateGraph() {
    if (!simData || !simulationGraph) return;

    const highlightChanged = document.getElementById('toggleChangedMinds').checked;
    
    // Get survey results for current step
    const currentSurvey = simData.survey_results.find(s => s.step === currentStep);
    
    // Get survey results for previous step
    const currentIdx = surveySteps.indexOf(currentStep);
    const prevStep = currentIdx > 0 ? surveySteps[currentIdx - 1] : null;
    const prevSurvey = prevStep !== null ? simData.survey_results.find(s => s.step === prevStep) : null;

    // Get all unique answers to assign colors
    const allAnswers = Array.from(new Set(simData.survey_results.flatMap(s => Object.values(s.results))));
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10).domain(allAnswers);

    d3.selectAll('.node')
        .attr('fill', d => {
            if (d.type === 'NewsSource') return '#000'; // Black for news sources
            
            if (!currentSurvey || !currentSurvey.results[d.name]) return '#ccc';
            
            const currentAns = currentSurvey.results[d.name];
            
            if (highlightChanged) {
                if (!prevSurvey || !prevSurvey.results[d.name]) return '#ccc';
                const prevAns = prevSurvey.results[d.name];
                if (currentAns !== prevAns) {
                    return colorScale(currentAns); // Color if changed
                } else {
                    return '#eee'; // Gray out if not changed
                }
            } else {
                return colorScale(currentAns);
            }
        })
        .attr('stroke', d => d === selectedNode ? '#000' : '#fff')
        .attr('stroke-width', d => d === selectedNode ? 3 : 1.5);
        
    if (selectedNode) {
        selectNode(selectedNode); // Update node details for current step
    }
}

function selectNode(node) {
    selectedNode = node;
    
    d3.selectAll('.node')
        .attr('stroke', d => d === selectedNode ? '#000' : '#fff')
        .attr('stroke-width', d => d === selectedNode ? 3 : 1.5);

    const infoDiv = document.getElementById('node-info');
    infoDiv.innerHTML = `
        <strong>Name:</strong> ${node.name}<br>
        <strong>Type:</strong> ${node.type}<br>
        <strong>Model ID:</strong> ${node.model_id || 'N/A'}<br>
        <strong>Degree:</strong> ${node.degree}
    `;

    // Find threads seen by this node up to currentStep
    const seenThreads = new Set();
    simData.observations.forEach(obs => {
        if (obs.entity_name === node.name && obs.step <= currentStep) {
            seenThreads.add(obs.thread_id);
        }
    });

    const threadsDiv = document.getElementById('node-threads');
    threadsDiv.innerHTML = `<h3>Threads Seen (${seenThreads.size})</h3>`;
    const threadsList = document.createElement('div');
    threadsList.style.maxHeight = '150px';
    threadsList.style.overflowY = 'auto';
    
    Array.from(seenThreads).slice(0, 20).forEach(tid => {
        const div = document.createElement('div');
        div.className = 'thread-item';
        div.innerText = `Thread ${tid}`;
        threadsList.appendChild(div);
    });
    if (seenThreads.size > 20) {
        const div = document.createElement('div');
        div.innerText = `... and ${seenThreads.size - 20} more`;
        threadsList.appendChild(div);
    }
    threadsDiv.appendChild(threadsList);

    // Find messages written by this node up to currentStep
    const writtenMessages = [];
    simData.threads.forEach(t => {
        if (t.messages) {
            t.messages.forEach(m => {
                if (m.role === node.name && m.step <= currentStep) {
                    writtenMessages.push({thread_id: t.id, content: m.content, step: m.step});
                }
            });
        }
    });

    const msgsDiv = document.getElementById('node-messages');
    msgsDiv.innerHTML = `<h3>Messages Written (${writtenMessages.length})</h3>`;
    const msgsList = document.createElement('div');
    msgsList.style.maxHeight = '200px';
    msgsList.style.overflowY = 'auto';
    
    writtenMessages.sort((a, b) => b.step - a.step).slice(0, 20).forEach(m => {
        const div = document.createElement('div');
        div.className = 'message-item';
        div.innerHTML = `<strong>Step ${m.step} (Thread ${m.thread_id}):</strong> ${m.content}`;
        msgsList.appendChild(div);
    });
    if (writtenMessages.length > 20) {
        const div = document.createElement('div');
        div.innerText = `... and ${writtenMessages.length - 20} more`;
        msgsList.appendChild(div);
    }
    msgsDiv.appendChild(msgsList);
}
