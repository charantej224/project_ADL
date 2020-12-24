var map = L.map('map').setView([39.0997, -94.5786], 10);

L.tileLayer('https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}?access_token=pk.eyJ1IjoibWFwYm94IiwiYSI6ImNpejY4NXVycTA2emYycXBndHRqcmZ3N3gifQ.rJcFIG214AriISLbB6B5aw', {
	maxZoom: 18,
	attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, ' +
		'<a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, ' +
		'Imagery © <a href="https://www.mapbox.com/">Mapbox</a>',
	id: 'mapbox/light-v9',
	tileSize: 512,
	zoomOffset: -1
}).addTo(map);


// control that shows state info on hover
var info = L.control();

info.onAdd = function (map) {
	this._div = L.DomUtil.create('div', 'info');
	this.update();
	return this._div;
};

info.update = function (props) {
	this._div.innerHTML = '<h4>311 Cases</h4>' +  (props ?
		'<b>' + props['ZCTA5CE10'] + '</b><br />' + (cases[props['ZCTA5CE10']] || 0) + ' cases.'
		: 'Hover over a Zip Code');
};

info.addTo(map);


// get color depending on population density value
function getColor(d) {
	return d > 50000  ? '#E31A1C' :
			d > 40000  ? '#FC4E2A' :
			d > 30000   ? '#FD8D3C' :
			d > 20000   ? '#FEB24C' :
			d > 10000   ? '#FED976' :
						'#FFEDA0';
// d > 1000 ? '#800026' :
//			d > 500  ? '#BD0026' :
			
}

function style(feature) {
	return {
		weight: 2,
		opacity: 1,
		color: 'white',
		dashArray: '3',
		fillOpacity: 0.7,
		fillColor: getColor(cases[feature.properties.ZCTA5CE10] || 0)
	};
}

function highlightFeature(e) {
	var layer = e.target;

	layer.setStyle({
		weight: 5,
		color: '#666',
		dashArray: '',
		fillOpacity: 0.7
	});

	if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
		layer.bringToFront();
	}

	info.update(layer.feature.properties);
}

var geojson;

function resetHighlight(e) {
	geojson.resetStyle(e.target);
	info.update();
}

function zoomToFeature(e) {
	map.fitBounds(e.target.getBounds());
	let zip_code = e.target.feature.properties['ZCTA5CE10']
	var categoriesChart = new CanvasJS.Chart("categories", {
		animationEnabled: true,
		title: {
			text: "Categories"
		},
		data: [{
			type: "pie",
			startAngle: 240,
			yValueFormatString: "##0 \"Cases\"",
			indexLabel: "{label} {y}",
			dataPoints: categories[zip_code]
		}]
	});
	categoriesChart.render();
	
	/* var timeSeriesChart = new CanvasJS.Chart("timeSeries", {
		animationEnabled: true,
		theme: "light2", // "light1", "light2", "dark1", "dark2"
		title:{
			text: "Time Series"
		},
		axisY: {
			title: "Cases"
		},
		data: [{        
			type: "column",  
			showInLegend: true, 
			legendMarkerColor: "grey",
			legendText: "Resolution Time in Hrs",
			dataPoints: [      
				{ y: 87.8, label: "2008" },
				{ y: 66.4,  label: "2009" },
				{ y: 97.09,  label: "2010" },
				{ y: 84,  label: "2011" },
				{ y: 75,  label: "2012" },
				{ y: 101.5, label: "2013" },
				{ y: 97.8,  label: "2014" },
				{ y: 80,  label: "2015" },
				{ y: 75,  label: "2016" },
				{ y: 60,  label: "2017" },
				{ y: 50,  label: "2018" },
				{ y: 40,  label: "2019" },
				{ y: 20,  label: "2020" },
			]
		}]
	});
	timeSeriesChart.render(); */
	
	var timeSeriesChart = new CanvasJS.Chart("timeSeries", {
		animationEnabled: true,  
		title:{
			text: "Time Series"
		},
		axisY: {
			title: "Avg. Resolution Time",
			valueFormatString: "#00",
			suffix: "hours"
		},
		data: [{
			type: "splineArea",
			color: "rgba(54,158,173,.7)",
			markerSize: 5,
			xValueFormatString: "YYYY",
			yValueFormatString: "$#,##0.##",
			dataPoints: [
				{ x: new Date(2008, 0), y: 87.8 },
				{ x: new Date(2009, 0), y: 66.4 },
				{ x: new Date(2010, 0), y: 97.09 },
				{ x: new Date(2011, 0), y: 84 },
				{ x: new Date(2012, 0), y: 75 },
				{ x: new Date(2013, 0), y: 60.5 },
				{ x: new Date(2014, 0), y: 50.8 },
				{ x: new Date(2015, 0), y: 80 },
				{ x: new Date(2016, 0), y: 75 },
				{ x: new Date(2017, 0), y: 60 },
				{ x: new Date(2018, 0), y: 50 },
				{ x: new Date(2019, 0), y: 40 },
				{ x: new Date(2020, 0), y: 20 }
			]
		}]
		});
		
	timeSeriesChart.render();
	
	let data = descriptions[zip_code];
	$('table#cases tbody').html('');
	
	for(let i = 0; i < data.length; i++) {
		let temp = '<tr><td><a href="http://city.kcmo.org/kc/ActionCenterRequest/CaseInfo.aspx?CaseID=' + data[i]['CASE ID'] + '" target="_blank">' + data[i]['CASE ID'] + '</a></td>';
		temp+= '<td>' + data[i]['DESCRIPTION']+ '</td>';
		$('table#cases tbody').append(temp);
	}
	$('#predictions p').hide()
	$('#predictions table').show()

}

function onEachFeature(feature, layer) {
	layer.on({
		mouseover: highlightFeature,
		mouseout: resetHighlight,
		click: zoomToFeature
	});
}

geojson = L.geoJson(zipData, {
	style: style,
	onEachFeature: onEachFeature
}).addTo(map);

map.attributionControl.addAttribution('Population data &copy; <a href="http://census.gov/">US Census Bureau</a>');


var legend = L.control({position: 'bottomright'});

legend.onAdd = function (map) {

	var div = L.DomUtil.create('div', 'info legend'),
		grades = [0, 10000, 20000, 30000, 40000, 50000],
		labels = [],
		from, to;

	for (let i = 0; i < grades.length; i++) {
		from = grades[i];
		to = grades[i + 1];

		labels.push(
			'<i style="background:' + getColor(from + 1) + '"></i> ' +
			from + (to ? '&ndash;' + to : '+'));
	}

	div.innerHTML = labels.join('<br>');
	return div;
};

legend.addTo(map);