import * as d3 from "d3";
import { RefObject, useEffect, useRef } from "react";
import { Dataset } from "../pages";
import Script from "next/script";

interface ScatterPlotProps {
  data: Dataset;
  width: number;
  height: number;
}

const removeSVGElements = (...classesOrIds: string[]) => {
  classesOrIds.forEach((name) => {
    d3.selectAll(name).remove();
  });
};

const drawScatterPlot = (
  svgRef: RefObject<SVGSVGElement>,
  data: Dataset,
  width: number,
  height: number
) => {
  // Clear previous plot's elements if any
  removeSVGElements(".plot-axis", ".plot-points", "#tooltip");

  // Prepare data
  // Get max value of X.slice([null, 0]) and X.slice([null, 1])
  const maxX0 = d3.max(data.X.slice([null, 0]).array() as number[])!;
  const minX0 = d3.min(data.X.slice([null, 0]).array() as number[])!;
  const maxX1 = d3.max(data.X.slice([null, 1]).array() as number[])!;
  const minX1 = d3.min(data.X.slice([null, 1]).array() as number[])!;
  const rangeX0 = maxX0 - minX0;
  const rangeX1 = maxX1 - minX1;

  const featureArray = data.X.array() as number[][];
  const labelArray = data.y.array() as number[];

  const marginAll = 50;
  const inner_width = width - marginAll * 2;
  const inner_height = height - marginAll * 2;

  const svg = d3
    .select(svgRef.current)
    .attr("width", width)
    .attr("height", height);

  // Add axis labels and plot title
  if (d3.selectAll(".axis-label").size() === 0) {
    svg
      .append("foreignObject")
      .attr("class", "axis-label")
      .attr("x", inner_width / 2 + marginAll)
      .attr("y", inner_height + marginAll * 1.1)
      .attr("width", 50)
      .attr("height", 25)
      .text("$$x_0$$")
      .style("font-size", "14px");

    svg
      .append("foreignObject")
      .attr("class", "axis-label")
      .attr("width", 50)
      .attr("height", 25)
      .text("$$x_1$$")
      .attr(
        "transform",
        `translate(${marginAll * 0.1}, ${
          inner_height / 2 + marginAll
        })rotate(-90)`
      )
      .style("font-size", "14px");
  }

  if (d3.selectAll(".plot-title").size() === 0) {
    svg
      .append("text")
      .attr("class", "plot-title")
      .attr("x", inner_width / 2 + marginAll)
      .attr("y", marginAll)
      .attr("text-anchor", "middle")
      .style("font-size", "20px")
      .text("Scatter Plot");
  }

  // Plotting code
  const xScale = d3
    .scaleLinear()
    .domain([minX0 - rangeX0 * 0.25, maxX0 + rangeX1 * 0.25]) // get max value of X.slice([:, 0])
    .range([0, inner_width]);

  const yScale = d3
    .scaleLinear()
    .domain([minX1 - rangeX1 * 0.25, maxX1 + rangeX1 * 0.25]) // get max value of X.slice([:, 1])
    .range([inner_height, 0]);

  const color = d3.scaleOrdinal(d3.schemeCategory10);
  const tooltip = d3
    .select("#plot-tooltip")
    .append("div")
    .attr("id", "tooltip")
    .style("position", "absolute")
    .style("visibility", "hidden")
    .style("z-index", "100") // Make sure tooltip is on top of everything
    .attr(
      "class",
      "p-2 text-sm font-normal text-gray-900 bg-white border border-gray-200 rounded-lg shadow-lg"
    );

  svg
    .append("g")
    .attr("class", "plot-axis")
    .attr("transform", `translate(${marginAll}, ${inner_height + marginAll})`)
    .call(d3.axisBottom(xScale));

  svg
    .append("g")
    .attr("class", "plot-axis")
    .attr("transform", `translate(${marginAll}, ${marginAll})`)
    .call(d3.axisLeft(yScale));

  svg
    .append("g")
    .attr("class", "plot-points")
    .selectAll("dot")
    .data(featureArray)
    .enter()
    .append("circle")
    .attr("cx", (d) => xScale(d[0]))
    .attr("cy", (d) => yScale(d[1]))
    .attr("r", 3)
    .style("fill", (d, i) => color(labelArray[i].toString()))
    .attr("transform", `translate(${marginAll}, ${marginAll})`)
    .on("mouseover", (ev, d) => {
      tooltip
        .style("left", `${ev.pageX + 10}px`)
        .style("top", `${ev.pageY + 10}px`)
        .style("visibility", "visible")
        .html(`( ${d[0].toFixed(3)}, ${d[1].toFixed(3)} )`);
    })
    .on("mouseout", () => {
      tooltip.style("visibility", "hidden");
    });
};

const ScatterPlot: React.FC<ScatterPlotProps> = ({ data, height, width }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;
    drawScatterPlot(svgRef, data, width, height);
  }, [data, height, width]);

  return (
    <>
      <Script
        id="MathJax-script"
        async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
      />
      <div className="outline-dashed outline-1" id="plot-tooltip">
        <svg ref={svgRef} width={width} height={height} />
      </div>
    </>
  );
};

export default ScatterPlot;
